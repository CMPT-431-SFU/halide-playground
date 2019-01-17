#include "halide_benchmark.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "gaussian_pipe.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

int main(int argc, char **argv) {

  if (argc < 2) {
    printf("Usage: ./run in.png \n");
    return 0;
  }

  Buffer<uint8_t> input = load_image(argv[1]);
  Buffer<uint8_t> out(input.width() - 8, input.height() - 8);

  printf("start.\n");

  gaussian_pipe(input, out);
  save_image(out, "out.png");

  printf("finish running native code\n");
  return 0;
}
