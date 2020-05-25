# Nonlinear Optimization
## Introduction
This project shows how to use least squares method to fit curve.
We use GaussNewton method, Ceres and g2o optimization packages to achieve it.

## Requirements
### OpenCV
#### Required Packages
OpenCV  
OpenCV Contrib

### Eigen Package (Version >= 3.0.0)
#### Source
http://eigen.tuxfamily.org/index.php?title=Main_Page

#### Compile and Install
```
cd [path-to-Eigen]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

#### Search Installing Location
```
sudo updatedb
locate eigen3
```

default location "/usr/include/eigen3"



### g2o Package
#### Download
https://github.com/RainerKuemmerle/g2o

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

### Ceres Package
#### Download
https://github.com/ceres-solver/ceres-solver

#### Required Dependences
```
sudo apt-get install liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev
```
Note:  
If the installation is failed, please search the non-installed package by Google, 
download and install it. Some packages have been saved in requiredPackages file.

#### Compile and Install
```
cd [path-to-pangolin]
mkdir build
cd build
cmake ..
make 
sudo make install 
```

## Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

## Run
### GaussNewton
```
./gaussNewton
```
### Ceres
```
./ceresCurceFitting
```
### g2o
```
./g2oCurveFitting
```

## Reference
[Source](https://github.com/HugoNip/slambook2/tree/master/ch6)
