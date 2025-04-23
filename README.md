# Parallel 3D Volume Rendering with MPI

**Implementation of distributed ray casting for volume rendering using 3D domain decomposition and MPI.**


## 🚀 Core Features
- **Distributed 3D Domain Decomposition**  
  - Partitions scalar volumetric data (`Isabel_High_Resolution_Raw`) across MPI processes along **X/Y/Z dimensions**.
  - Ensures balanced workload distribution for scalable rendering.

- **Distributed Ray Casting**  
  - Each MPI process performs **local ray casting** on its subdomain.
  - Supports **front-to-back compositing** with configurable step size.

- **Distributed Image Composition**  
  - Implements **parallel binary-swap compositing** to merge partial images across processes.
  - Minimizes communication overhead during final image stitching.

- **Performance-Optimized**  
  - Tracks **distributed timing metrics**: computation (local rendering), communication (MPI sync), and total runtime.
  - Outputs rendered image (`PX_PY_PZ.png`) and performance logs.
