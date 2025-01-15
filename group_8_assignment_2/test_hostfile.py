# code.py
from mpi4py import MPI
import socket

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get hostname of each process
hostname = socket.gethostname()

print(f"Hello from rank {rank} out of {size} on host {hostname}")

# Finalize MPI (optional in simple scripts, but good practice)
MPI.Finalize()
