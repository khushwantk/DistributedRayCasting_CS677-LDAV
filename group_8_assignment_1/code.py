# Tasks :
# 1. File read by rank 0   [Done]
# 1.1. Crop the dataset acc to xy min,max  [Done]
# 2. Data Decomposition acc to type and Rank 0 distributes sliced data to all ranks [Done]
# 3. RayCasting in parallel by each rank on assigned slice[Done]
# 3. Output sub-image by each rank  [Done]
# 5. Stitch the image and generate a png  [Done]

from mpi4py import MPI
import numpy as np
import time
import sys
import math
import matplotlib.pyplot as plt

def get_shape_from_dataset_name(dataset_name):
    if "1000x1000x200" in dataset_name:
        return (1000, 1000, 200)
    elif "2000x2000x400" in dataset_name:
        return (2000, 2000, 400)
    else:
        raise ValueError("Dataset name doesn't match known dimensions.")

def load_data(filename, shape):
    data = np.fromfile(filename, dtype=np.float32).reshape(shape, order='F')
    return data

def slicing_t1(num_slices, data):
    total_rows = data.shape[0]
    slice_size = total_rows // num_slices
    extra_rows  = total_rows % num_slices
    split_indices = [slice_size * i + min(i, extra_rows) for i in range(1, num_slices)]
    return np.split(data, split_indices, axis=0)

def find_optimal_slices(num_slices):
    for i in range(int(math.sqrt(num_slices)), 0, -1):
        if num_slices % i == 0:
            x_slices = i
            y_slices = num_slices // i
            return max(x_slices, y_slices), min(x_slices, y_slices)
    return num_slices, 1

def slicing_t2(num_slices, data):
    y_slices,x_slices = find_optimal_slices(num_slices)
    y_splits = np.array_split(data, y_slices, axis=1)
    final_splits = [np.array_split(part, x_slices, axis=0) for part in y_splits]
    final_splits = [subarray for split in final_splits for subarray in split]
    return final_splits

# def extract_region(image, x_min, x_max, y_min, y_max):
#     return image[max(0, x_min):min(image.shape[0], x_max),max(0, y_min):min(image.shape[1], y_max), :]

def load_transfer_function(file_path, is_color=True):
    with open(file_path, 'r') as file:
        # Read and clean all lines in a single pass
        data = [float(val) for line in file for val in line.strip().replace(',', '').split()]

    if is_color:
        return [(data[i], tuple(data[i+1:i+4])) for i in range(0, len(data), 4)]
    else:
        return [(data[i], data[i+1]) for i in range(0, len(data), 2)]


def interpolate_color(query_val, color_tf):
    for i in range(len(color_tf) - 1):
        x0, rgb0 = color_tf[i]
        x1, rgb1 = color_tf[i+1]
        if x0 <= query_val <= x1:
            t = (query_val - x0) / (x1 - x0)
            return [rgb0[j] + t * (rgb1[j] - rgb0[j]) for j in range(3)]
    return [0, 0, 0]

def interpolate_color2(query_val, color_tf):
    x = np.array([point[0] for point in color_tf])
    r = np.array([point[1][0] for point in color_tf])
    g = np.array([point[1][1] for point in color_tf])
    b = np.array([point[1][2] for point in color_tf])

    r_interp = np.interp(query_val, x, r)
    g_interp = np.interp(query_val, x, g)
    b_interp = np.interp(query_val, x, b)
    return [r_interp, g_interp, b_interp]

def interpolate_value(data, x, y, z):
    z0 = int(z)
    z1 = min(z0 + 1, data.shape[2] - 1)
    t = z - z0
    return (1 - t) * data[x, y, z0] + t * data[x, y, z1]

def interpolate_value2(data, x, y, z):
    z0 = int(np.floor(z))
    z1 = min(z0 + 1, data.shape[2] - 1)
    return np.interp(z, [z0, z1], [data[x, y, z0], data[x, y, z1]])

def interpolate_opacity(query_val, opacity_tf):
    for i in range(len(opacity_tf) - 1):
        x0,y0=opacity_tf[i]
        x1,y1=opacity_tf[i+1]
        if x0 <= query_val <= x1:
            return y0+((y1-y0)*(query_val-x0)/(x1-x0))

def interpolate_opacity2(query_val, opacity_tf):
    x = np.array([point[0] for point in opacity_tf])
    y = np.array([point[1] for point in opacity_tf])
    return np.interp(query_val, x, y)

def ray_casting(subdomain, opacity_points, color_points, step_size,rank):
    height, width, depth = subdomain.shape
    img = np.zeros((height, width,3))
    terminated_rays = 0

    for y in range(width):
        for x in range(height):
            accumulated_color = np.zeros(3)
            accumulated_opacity = 0
            z = 0.0
            while z < depth:

                data_val = interpolate_value(subdomain,x,y,z)
                color = np.array(interpolate_color(data_val, color_points))
                opacity = interpolate_opacity(data_val, opacity_points)

                accumulated_color += (1 - accumulated_opacity) * color * opacity
                accumulated_opacity +=  (1 - accumulated_opacity) * opacity

                if accumulated_opacity >= 0.98:
                    terminated_rays += 1
                    break

                z += step_size

            img[x, y,:] = accumulated_color

    return img,terminated_rays

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if len(sys.argv) < 8:
        if rank == 0:
            print("Usage: mpirun -np <num_procs> python file_name.py <dataset_name> <decomposition_type> <step_size> <x_min> <x_max> <y_min> <y_max>")
        sys.exit()

    dataset_name = sys.argv[1]
    decomposition_type = int(sys.argv[2])
    step_size = float(sys.argv[3])
    x_min = int(sys.argv[4])
    x_max = int(sys.argv[5])
    y_min = int(sys.argv[6])
    y_max = int(sys.argv[7])


    shape = get_shape_from_dataset_name(dataset_name)

    if rank == 0:
        start_time = time.time()
        data = load_data(dataset_name, shape)

        # Cropping entire dataset based on bounds
        x_min = max(x_min, 0)
        x_max = min(x_max, data.shape[1])
        y_min = max(y_min, 0)
        y_max = min(y_max, data.shape[0])
        data = data[x_min:x_max, y_min:y_max, :]
        new_shape=data.shape

        if(decomposition_type==1):
            data =slicing_t1(size,data)
        elif(decomposition_type==2):
            data =slicing_t2(size,data)

        elapsed_time = time.time() - start_time

        print(f"Time taken for data loading and decomposition according to type {decomposition_type} : {elapsed_time:.4f}sec")

        # print("Shapes of all subdomains after decomposition: ", end="")
        # for i in range(len(data)):
        #     print(data[i].shape, end=" ")
        # print()


        # Data Distribution
        for i in range(1, size):
            comm.send(data[i], dest=i, tag=i)
            # print(f"Rank {rank} has sent subdomain {i}")

        subdomain=data[0]

    else:
        subdomain = comm.recv(source=0, tag=rank)


    # print(f"Subdomian assigned to rank {rank} is of shape: {subdomain.shape}")


    color_tf_filename = 'color_TF.txt'
    opacity_tf_filename = 'opacity_TF.txt'
    opacity_points = load_transfer_function(opacity_tf_filename, is_color=False)
    color_points = load_transfer_function(color_tf_filename, is_color=True)


    start_time2 = MPI.Wtime()
    img, terminated_rays = ray_casting(subdomain,opacity_points, color_points, step_size,rank)
    end_time2 = MPI.Wtime()
    # print(f"Rank {rank} finished ray casting")

    gathered_images = comm.gather(img, root=0)
    rays_terminated_early_total = comm.reduce(terminated_rays, op=MPI.SUM, root=0)



    if rank == 0:
        # print(f"Time taken for data loading and decomposition to type {decomposition_type} : ",elapsed_time)
        print(f"Time taken for parallel ray casting : {end_time2 - start_time2 :.4f}sec or {((end_time2 - start_time2)/60):.4f}  min")

        if decomposition_type == 1:
            final_image = np.vstack(gathered_images)
        else:
            y_slices,x_slices = find_optimal_slices(size)
            columns = []
            for j in range(y_slices):
                col_blocks = gathered_images[j * x_slices:(j + 1) * x_slices]
                col = np.vstack(col_blocks)
                columns.append(col)
                # print("Shape of stacked column: ", col.shape)
            final_image = np.hstack(columns)

        file_name=f"bounded_{size}_{decomposition_type}_{step_size}.png"
        plt.imsave(file_name,final_image)

        total_rays = new_shape[0] * new_shape[1]
        fraction_early_terminated = rays_terminated_early_total / total_rays
        print(f'Fraction of rays terminated early: {rays_terminated_early_total} / {total_rays}= {fraction_early_terminated:.4f}')

if __name__ == "__main__":
    main()
