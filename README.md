# halide-playground

Group of halide apps demonstrating generators, runtime, and bitcode creation.

## Dependencies
Halide library and runtime, LLVM-IR-CMake Utils, -libjpeg, -libpng, any modern C++11 compiler
```
sudo apt-get install libjpeg-dev libpng-dev 
```

## Build
```
mkdir build; cd build; 
cmake ../ -DHALIDE_DISTRIB_DIR=PATH_OF_DISTRIB_DIR
make 
```
