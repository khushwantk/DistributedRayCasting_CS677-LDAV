Score : 104/110

We have used mpi4py, numpy and matplotlib libraries.

To install we have created a install_libraries.sh

If that doesn't work do this manually :

```
pip install mpi4py
pip install numba numpy matplotlib
```


You can check if the all hosts from hostfile are working by running :

`mpirun --oversubscribe --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 --oversubscribe python3 test_hostfile.py`

To run our main code, (first ensure all hosts are working)
1. Place the dataset,color_Tf.txt,opacity_TF.txt in the same folder as A2.py
2. Use the following command to run our code on smaller dataset:

    `mpirun --oversubscribe --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 python3 A2.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt`

    `mpirun --oversubscribe --mca btl_tcp_if_include eno1 --hostfile hostfile -np 16 python3 A2.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt`

    `mpirun --oversubscribe --mca btl_tcp_if_include eno1 --hostfile hostfile -np 32 python3 A2.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt`


Use python or python3 according to what is installed.

Output will be a file named `{PX}_{PY}_{PZ}.png`

Images used in the report are reduced to their 50% height and width.

Original rendered Images and terminal outputs are available in output folder.
Inside the output folder we have also attached the exact terminal screen output.txt files and screenshots too.

Plots attached in the report are generated in plots.ipynb file.

**Please ensure that mpirun and python mpi4py are both of same MPI ie. mpi4py should have been compiled from same MPI installed on system either OpenMPI or MPICH.**
