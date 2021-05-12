## Installation

### Prerequisites

- Ubuntu 16.04+
- Python 3.6+
- NumPy 1.19
- PyTorch (tested on 1.5.0/1.7.1)
- [MMCV](https://github.com/open-mmlab/mmcv) (tested on 1.0.5/1.2.1)
- [MMDetection](https://github.com/open-mmlab/mmdetection) (tested on 2.3.0/2.7.0)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) (tested on 0.6.0/0.8.0)

### Install Ceres 1.14

Note: These instructions are for Ubuntu 18.04. On Ubuntu 16.04, some dependencies (e.g. GFlags) may need to be manually installed.

```bash
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse
sudo apt-get install libsuitesparse-dev
# build Ceres from source (without installation)
wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar zxf ceres-solver-1.14.0.tar.gz
cd ceres-solver-1.14.0
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j8
cd ..
export CERES_INCLUDE_DIRS=$PWD/include
export Ceres_DIR=$PWD/build
```

### Clone and build

Clone this repo to `$MonoRUn_ROOT` and build the PnP module:

```bash
# build PnP module
cd $MonoRUn_ROOT/monorun/ops/least_squares
python setup.py
export LD_LIBRARY_PATH=$Ceres_DIR/lib:$LD_LIBRARY_PATH
# alternatively you can add LD_LIBRARY_PATH into .bashrc
echo "export LD_LIBRARY_PATH=$Ceres_DIR/lib:"'$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```