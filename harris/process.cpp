#include "halide_benchmark.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "harris_pipe.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

int main(int argc, char **argv)
{
  // float k = 0.04;
  // float threshold = 100;

  Buffer<uint8_t> input = load_and_convert_image(argv[1]);
  Buffer<uint8_t> out_native(input.width() - 6, input.height() - 6);

  printf("start.\n");

  harris_pipe(input, out_native);
  save_image(out_native, "out.png");

  printf("finished running native code\n");
}
