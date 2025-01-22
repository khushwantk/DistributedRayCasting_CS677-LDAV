Score : 110/110

We have used mpi4py, numpy and matplotlib libraries.

To install we have created a install_libraries.sh

If that doesn't work :
```
pip install mpi4py
pip install numpy
pip install matplotlib
```

To run our code,
1. Place the dataset,color_Tf.txt,opacity_TF.txt in the same folder as code.py
2.  Use the following command to run our code :

    `mpirun -np 4 python3 code.py Isabel_1000x1000x200_float32.raw 1 0.75 0 999 0 1000`

    `mpirun -np 8 python3 code.py Isabel_1000x1000x200_float32.raw 2 0.25 0 500 0 999`

    `mpirun -np 15 python3 code.py Isabel_1000x1000x200_float32.raw 1 0.5 0 999 0 700`

    `mpirun -np 32 python3 code.py Isabel_1000x1000x200_float32.raw 2 0.35 0 500 0 999`

Use python or python3 according to what is installed.

Output will be a file named `bounded_{np}_{type}_{step_size}`

Images used in the report are reduced to their 50% height and width.

Original render Images are available in images folder.

Folder (images/1/) for the test examples.

Folder (images/2/) for the our sample run on bigger dataset.

Folder (images/full/) contains the full data extent render.
