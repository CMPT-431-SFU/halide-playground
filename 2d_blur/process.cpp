// Download a Halide distribution from halide-lang.org and untar it in
// the current directory. Then you should be able to compile this
// file with:
//
// c++ -g blur.cpp -std=c++11 -I$HALIDE_DISTRIB_DIR -L halide/bin/ -lHalide
// `libpng-config --cflags
// --ldflags` -O3
// g++ process.cpp - g - I ${HALIDE_DISTRIB_DIR} / include - I
// ${HALIDE_DISTRIB_DIR} / tools - L ${HALIDE_DISTRIB_DIR} / bin - lHalide
// `libpng - config-- cflags-- ldflags` - ljpeg - lpthread - ldl - o
// 2dblur_process - std = c++ 11
//
// You'll also need a multi-megapixel png image to run this on. Name
// it input.png and put it in this directory.

// Include the Halide language
#include "Halide.h"
using namespace Halide;
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"
#include <iostream>

// Some support code for timing and loading/saving images
#include <chrono>

using namespace Halide;
using namespace Halide::Tools;
using namespace std;

int main(int argc, char **argv)
{
  // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
  float sigma = 1.5f;
  // Take a color 8-bit input
  Var x("x"), y("y"), c("c");

  Halide::Buffer<uint8_t> input = load_image(argv[1]);

  // This time, we'll wrap the input in a Func that prevents
  // reading out of bounds:
  Func clamped("clamped");

  // Define an expression that clamps x to lie within the
  // range [0, input.width()-1].
  Expr clamped_x = clamp(x, 0, input.width() - 1);
  // clamp(x, a, b) is equivalent to max(min(x, b), a).

  // Similarly clamp y.
  Expr clamped_y = clamp(y, 0, input.height() - 1);
  // Load from input at the clamped coordinates. This means that
  // no matter how we evaluated the Func 'clamped', we'll never
  // read out of bounds on the input. This is a clamp-to-edge
  // style boundary condition, and is the simplest boundary
  // condition to express in Halide.
  clamped(x, y, c) = input(clamped_x, clamped_y, c);

  // Defining 'clamped' in that way can be done more concisely
  // using a helper function from the BoundaryConditions
  // namespace like so:
  //
  // clamped = BoundaryConditions::repeat_edge(input);
  //
  // These are important to use for other boundary conditions,
  // because they are expressed in the way that Halide can best
  // understand and optimize. When used correctly they are as
  // cheap as having no boundary condition at all.

  // Upgrade it to 16-bit, so we can do math without it
  // overflowing. This time we'll refer to our new Func
  // 'clamped', instead of referring to the input image
  // directly.
  Func input_16("input_16");
  input_16(x, y, c) = cast<uint16_t>(clamped(x, y, c));

  // The rest of the pipeline will be the same...

  // Blur it horizontally:
  Func blur_x("blur_x");
  blur_x(x, y, c) =
      (input_16(x - 2, y, c) + input_16(x - 1, y, c) + input_16(x, y, c) + input_16(x + 1, y, c) + input_16(x + 2, y, c)) / 5;

  // Blur it vertically:
  Func blur_y("blur_y");
  blur_y(x, y, c) =
      (input_16(x, y - 2, c) + blur_x(x, y - 1, c) + blur_x(x, y, c) + blur_x(x, y + 1, c) + blur_x(x, y + 2, c)) / 5;

  // Convert back to 8-bit.
  Func output("output");
  Func output_x("output_x");
  output_x(x, y, c) = cast<uint8_t>(blur_x(x, y, c));
  output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));
  Buffer<uint8_t> result_x = output_x.realize(input.width(), input.height(), 3);
  Buffer<uint8_t> result = output.realize(input.width(), input.height(), 3);
  // Save the result. It should look like a slightly blurry
  // parrot, but this time it will be the same size as the
  // input.
  save_image(result_x, "xblur_" + (std::string)argv[2]);
  save_image(result, argv[2]); // Benchmark the pipeline.
  return 0;
}
