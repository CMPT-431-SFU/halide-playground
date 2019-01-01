# halide-playground

Group of halide apps demonstrating generators, runtime, and bitcode creation.

## Dependencies
Halide library and runtime, LLVM-IR-CMake Utils, -libjpeg, -libpng, any modern C++11 compiler
```
sudo apt-get install libjpeg-dev libpng-dev 
```

## Example: Generating bitcode and exe for wavelet.
```
cd wavelet; mkdir build; cd build; 
cmake ../ 
make 
make wavelet_bc
make All 
```