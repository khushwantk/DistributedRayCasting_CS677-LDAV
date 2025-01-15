from mpi4py import MPI
import numpy as np
import time
import sys
import warnings
import gc
from PIL import Image
import socket

from numba import jit,prange

warnings.simplefilter("ignore", UserWarning)
# import matplotlib.pyplot as plt


def pad_data(data, px, py):
    original_shape = data.shape

    # Calculate the target shape
    target_height = np.ceil(original_shape[0] / px).astype(int) * px
    target_width = np.ceil(original_shape[1] / py).astype(int) * py

    # Calculate padding widths
    pad_height = target_height - original_shape[0]
    pad_width = target_width - original_shape[1]

    # Pad heights and widths with (before, after) tuple
    pad_widths = [(0, pad_height), (0, pad_width), (0, 0)]  # No padding for z-axis

    # Pad the data
    padded_data = np.pad(data, pad_widths, mode='constant', constant_values=0)

    return padded_data, original_shape

def unpad_image(final_image, original_shape):
    # Remove padding based on the original shape
    return final_image[:original_shape[0], :original_shape[1], :]

def get_shape_from_dataset_name(dataset_name):
    if "1000x1000x200" in dataset_name:
        return (1000, 1000, 200)
    elif "2000x2000x400" in dataset_name:
        return (2000, 2000, 400)
    else:
        raise ValueError("Dataset doesn't match.")

def load_data(filename, shape,px,py):
    data=np.memmap(filename, dtype=np.float32, mode='r', shape=shape, order='F')
    padded_data, original_shape = pad_data(data, px, py)
    return padded_data, original_shape

def load_transfer_function(file_path, is_color=True):
    with open(file_path, 'r') as file:
        data = [float(val) for line in file for val in line.strip().replace(',', '').split()]

    if is_color:
        return [(data[i], data[i + 1], data[i + 2], data[i + 3]) for i in range(0, len(data), 4)]
    else:
        return [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

@jit(nopython=True)
def interpolate_color(query_val, color_tf):
    for i in range(len(color_tf) - 1):
        x0, rgb0_r, rgb0_g, rgb0_b = color_tf[i]
        x1, rgb1_r, rgb1_g, rgb1_b = color_tf[i + 1]
        if x0 <= query_val <= x1:
            t = (query_val - x0) / (x1 - x0)
            return np.array([rgb0_r + t * (rgb1_r - rgb0_r),
                             rgb0_g + t * (rgb1_g - rgb0_g),
                             rgb0_b + t * (rgb1_b - rgb0_b)], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0], dtype=np.float64)

@jit(nopython=True)
def interpolate_opacity(query_val, opacity_tf):
    for i in range(len(opacity_tf) - 1):
        x0, y0 = opacity_tf[i]
        x1, y1 = opacity_tf[i + 1]
        if x0 <= query_val <= x1:
            return y0 + ((y1 - y0) * (query_val - x0) / (x1 - x0))
    return 0

@jit(nopython=True)
def interpolate_value(data, x, y, z):
    z0 = int(z)
    z1 = min(z0 + 1, data.shape[2] - 1)
    t = z - z0
    return (1 - t) * data[x, y, z0] + t * data[x, y, z1]


@jit(nopython=True)
def ray_casting(subdomain, opacity_points, color_points, step_size):
    height, width, depth = subdomain.shape
    img = np.zeros((height, width, 4))
    for y in range(width):
        for x in range(height):
            accumulated_color = np.zeros(3)
            accumulated_opacity = 0
            z = 0.0
            while z < depth:
                data_val = interpolate_value(subdomain, x, y, z)
                color = interpolate_color(data_val, color_points)
                opacity = np.array(interpolate_opacity(data_val, opacity_points))

                accumulated_color += (1 - accumulated_opacity) * color * opacity
                accumulated_opacity += (1 - accumulated_opacity) * opacity

                if accumulated_opacity >= 0.98:
                    break

                z += step_size

            img[x, y, :3] = accumulated_color
            img[x, y, 3] = accumulated_opacity

    return img




def binary_swap_compositing(img, PX, PY, PZ, rank, comm):
    print(f"Rank {rank} has started merging...")
    sys.stdout.flush()
    num_ranks = comm.Get_size()
    sub_height, sub_width, _ = img.shape  # Dimensions of the subdomain image

    # Separate color and opacity channels
    local_composite_color = img[:, :, :3]
    local_composite_opacity = img[:, :, 3]

    total_send_recv_time = 0.0
    total_computation_time = 0.0

    # Determine the px, py, and pz coordinates for the current rank
    px = (rank // (PY * PZ)) % PX
    py = (rank // PZ) % PY
    pz = rank % PZ

    # Perform binary swap compositing only along the pz axis
    step = 1
    while step < PZ:
        partner_pz = pz ^ step  # XOR to find partner along the depth (pz) axis
        partner_rank = px * PY * PZ + py * PZ + partner_pz

        if partner_pz < PZ:
            recv_color = np.empty_like(local_composite_color)
            recv_opacity = np.empty_like(local_composite_opacity)


            if partner_rank > rank:
                # Start timing send operation
                start_send_time = MPI.Wtime()
                comm.Send(np.ascontiguousarray(local_composite_color), dest=partner_rank)
                comm.Send(np.ascontiguousarray(local_composite_opacity), dest=partner_rank)
                end_send_time = MPI.Wtime()
                total_send_recv_time += (end_send_time - start_send_time)

            else:
                # Receive data from partner rank
                start_recv_time = MPI.Wtime()
                comm.Recv(recv_color, source=partner_rank)
                comm.Recv(recv_opacity, source=partner_rank)
                end_recv_time = MPI.Wtime()
                total_send_recv_time += (end_recv_time - start_recv_time)

                # Perform the compositing operation
                start_comp_time = MPI.Wtime()
                remaining_opacity = 1 - local_composite_opacity
                local_composite_color += recv_color * remaining_opacity[:, :, None]
                # local_composite_opacity += recv_opacity
                # local_composite_opacity = np.clip(local_composite_opacity + recv_opacity, 0, 1)
                local_composite_opacity += recv_opacity * remaining_opacity
                local_composite_opacity = np.clip(local_composite_opacity, 0, 1)
                end_comp_time = MPI.Wtime()
                total_computation_time += (end_comp_time - start_comp_time)

        step *= 2  # Move to the next depth level

    # Gather results to rank 0 for the (px, py) layer
    start_gather_time = MPI.Wtime()
    gathered_colors = comm.gather(local_composite_color, root=0)
    gathered_opacities = comm.gather(local_composite_opacity, root=0)
    end_gather_time = MPI.Wtime()
    gather_time = end_gather_time - start_gather_time

    total_communication_time = total_send_recv_time + gather_time

    # Final image assembly on rank 0
    final_image = None
    merge_computation_time = 0.0

    if rank == 0:
        final_image = np.zeros((PX * sub_height, PY * sub_width, 3))
        total_opacity = np.zeros((PX * sub_height, PY * sub_width))

        start_merge_time = MPI.Wtime()

        for i in range(num_ranks):
            px = (i // (PY * PZ)) % PX
            py = (i // PZ) % PY
            pz = i % PZ


            # Place the composited subdomain in the final image grid for each (px, py)
            if(pz==PZ//2):
                final_image[px * sub_height:(px + 1) * sub_height,
                        py * sub_width:(py + 1) * sub_width, :] = gathered_colors[i]

                total_opacity[px * sub_height:(px + 1) * sub_height,
                          py * sub_width:(py + 1) * sub_width] += gathered_opacities[i]

        end_merge_time = MPI.Wtime()
        merge_computation_time = end_merge_time - start_merge_time

        total_computation_time += merge_computation_time

        return final_image, total_communication_time, total_computation_time

    else:
        return final_image, total_communication_time, total_computation_time




def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    for i in range(size):
        if rank == i:
            print(f"Rank {rank} running on {socket.gethostname()}")
            sys.stdout.flush()
        comm.Barrier()

    start_time_full = MPI.Wtime()  # Start timing the total execution

    # Initialize times for all ranks
    send_time = 0.0
    recv_time = 0.0
    merge_time = 0.0


    if len(sys.argv) < 8:
        if rank == 0:
            print("Usage: mpirun -np <num_procs> python file_name.py <dataset_name> <PX> <PY> <PZ> <step_size> <opacity_tf> <color_tf>")
        sys.exit()

    dataset_name = sys.argv[1]
    PX = int(sys.argv[2])
    PY = int(sys.argv[3])
    PZ = int(sys.argv[4])
    step_size = float(sys.argv[5])
    opacity_tf_filename = sys.argv[6]
    color_tf_filename = sys.argv[7]

    shape = get_shape_from_dataset_name(dataset_name)
    # print(shape)

    if rank == 0:

        start_time = MPI.Wtime()
        # Padded data if 1000/px, 1000/py . Same for 2000. Later removing the pad in final image
        data ,original_shape= load_data(dataset_name, shape,PX,PY)


        # Splitting data into subdomains based on PX, PY, PZ
        x_slices = np.array_split(data, PX, axis=0)
        subdomains = [np.array_split(slice, PY, axis=1) for slice in x_slices]

        # Distributing subdomains to all ranks
        start_send = MPI.Wtime()
        for i in range(PX):
            for j in range(PY):
                for k in range(PZ):
                    dest_rank = i * PY * PZ + j * PZ + k
                    to_send=subdomains[i][j][:, :, k::PZ]
                    if dest_rank == 0:
                        subdomain = to_send  # Keep this subdomain for rank 0
                    else:
                        # print(f"Sending subdomain of size {to_send.shape} to rank {dest_rank}")
                        comm.send(to_send, dest=dest_rank, tag=dest_rank)

        print("Domain decomposition and distribution finished.")
        sys.stdout.flush()

        end_send = MPI.Wtime()
        send_time = end_send - start_send

        delapsed_time = MPI.Wtime() - start_time
        # print(f"Time taken for data loading and distribution: {delapsed_time:.4f} seconds")

        # Run garbage collection to release memory
        del subdomains
        gc.collect()
        del x_slices
        gc.collect()
        del to_send
        gc.collect()
        del data
        gc.collect()


    else:
        start_recv = MPI.Wtime()
        subdomain = comm.recv(source=0, tag=rank)
        end_recv = MPI.Wtime()
        recv_time = end_recv - start_recv

    # print(f"Rank {rank} has subdomain of shape: {subdomain.shape}")


    opacity_points = load_transfer_function(opacity_tf_filename, is_color=False)
    color_points = load_transfer_function(color_tf_filename, is_color=True)

    start_time = MPI.Wtime()
    img = ray_casting(subdomain, opacity_points, color_points, step_size)
    end_time = MPI.Wtime()
    ray_computation_time = end_time - start_time

    print(f"Rank {rank} finished raycasting in {ray_computation_time:.4f} seconds.")
    sys.stdout.flush()


    final_image, merge_communication_time, merge_computation_time = binary_swap_compositing(img,PX,PY,PZ, rank, comm)


    computation_time_rank=merge_computation_time+ray_computation_time
    max_computation_time = comm.reduce(computation_time_rank, op=MPI.MAX, root=0)


    total_comm_time_rank = send_time + recv_time + merge_communication_time
    max_total_comm_time = comm.reduce(total_comm_time_rank, op=MPI.MAX, root=0)

    print(f"Time taken by rank {rank} on computations : {computation_time_rank:.4f} seconds")
    sys.stdout.flush()

    print(f"Time taken by rank {rank} on communications : {total_comm_time_rank:.4f} seconds")
    sys.stdout.flush()

    if rank == 0:
        filename = f"{PX}_{PY}_{PZ}.png"
        final_image=unpad_image(final_image,original_shape)
        # plt.imsave(filename, final_image)


        flipped_img = np.flipud(final_image)
        rotated_img = np.rot90(flipped_img)
        pillow_image = Image.fromarray((rotated_img * 255).astype(np.uint8))
        pillow_image.save(filename)

        print(" ")
        print(f"Final image saved as: {filename} with dimensions {final_image.shape}")
        print(f"Time taken for data loading and distribution: {delapsed_time:.4f} seconds")
        print(f"Max Computation time across all ranks: {max_computation_time:.4f} seconds")
        print(f"Max Communication time across all ranks: {max_total_comm_time:.6f} seconds")


    comm.Barrier()
    total_execution_time = MPI.Wtime() - start_time_full

    print(f"Time taken by rank {rank} to finish execution : {total_execution_time:.4f} seconds")
    sys.stdout.flush()
    full_time_max = comm.reduce(total_execution_time, op=MPI.MAX, root=0)
    full_time_min = comm.reduce(total_execution_time, op=MPI.MIN, root=0)

    global_time_sum=comm.reduce(total_execution_time, op=MPI.SUM, root=0)
    global_time_mean=None

    if rank == 0:
        global_time_mean=global_time_sum/size

    global_time_mean=comm.bcast(global_time_mean,root=0)
    local_sq_diff=np.sum((total_execution_time-global_time_mean)**2)
    global_sq_diff=comm.reduce(local_sq_diff,op=MPI.SUM,root=0)

    if rank == 0:
        global_var=global_sq_diff/size

        print(f"Difference between max and min total execution time across different ranks : {(full_time_max-full_time_min):.4f} seconds")
        print(f"Variance among total execution time of all ranks: {global_var:.5f}")
        print(f"Total execution time of code : {full_time_max:.4f} seconds")
        print(" ")
        sys.stdout.flush()

    #print(f"Time taken by rank {rank} to finish execution : {total_execution_time:.4f} seconds")
    #sys.stdout.flush()

    MPI.Finalize()

if __name__ == "__main__":
    main()






# mpirun --oversubscribe -np 8 python3 codee.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

# mpirun --oversubscribe --mca btl_tcp_if_include eno1 --hostfile hostfile -np 32 --oversubscribe python3 A2.py Isabel_2000x2000x400_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt
