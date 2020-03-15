// Download a Halide distribution from halide-lang.org and untar it in
// the current directory. Then you should be able to compile this
// file with:
//

// g++ process.cpp -g -I${HALIDE_DISTRIB_DIR}/include
// -I${HALIDE_DISTRIB_DIR}/tools -L${HALIDE_DISTRIB_DIR}/bin -lHalide
// `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o 2dblur_process
// -std=c++11

// You'll also need a multi-megapixel png image to run this on. Name
// it input.png and put it in this directory.

// Include the Halide language
#include "Halide.h"
using namespace Halide;
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "timing.h"
#include <iostream>

// Some support code for timing and loading/saving images
#include <chrono>

using namespace Halide;
using namespace Halide::Tools;
using namespace std;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
  // load the input, convert to single channel and turn into Halide Image
  Halide::Buffer<uint8_t> im = load_image(argv[1]);
  //  Image<uint8_t> im = load<uint8_t>("images/rgb.png");

  int width = im.width();
  int height = im.height();

  ////Declarations of Halide variable and function names

  Var x("x"), y("y"), c("c"); // declare domain variables
  Var xo("xo"), yo("yo"), xi("xi"),
      yi("yi"); // Declare the inner tile variables

  // 3 channel to 1 channel conversion: we should take luminance
  // but here we only want to test schedules, so it does not matter.
  Func input("input");
  input(x, y) = cast<float>(im(x, y, 0));

  float refTime = 0.0f;

  //////// SCHEDULE 1 : ROOT ////////
  {
    Func blur_x("blur_x"); // declare the horizontal blur function
    Func blur_y("blur_y"); // declare the vertical blur function
    blur_x(x, y) =
        cast<float>(input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3.0f;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

    // C++ style schedule: root. We compute the first stage (blur_x)
    // on the entire image before computing the second stage.
    blur_y.compute_root();
    blur_x.compute_root();
    // this schedule has bad locality because the data produced by blur_x
    // are long ejected from the cache by the time blur_y needs them
    // it also doesn"t have any parallelism
    // But it doesn"t introduce any extra computation

    // Finally compile and run.
    // Note that subtract two to the height and width to avoid boundary issues
    cout << "\nSchedule 1, ROOT:\t";
    float t = profile(blur_y, width - 2, height - 2);
    blur_y.compile_to_lowered_stmt("blur_schedule1.html", {}, Halide::HTML);
    refTime = t;
  }

  //////// SCHEDULE 2 : INLINE ////////
  {
    Func blur_x("blur_x"); // declare the horizontal blur function
    Func blur_y("blur_y"); // declare the vertical blur function
    blur_x(x, y) =
        cast<float>(input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3.0f;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

    // In this schedule, we compute values for blur_x each time they are
    // required by blur_y (inline). This means excellent locality between
    // producer and consumer, since blur_x values are produced as needed and
    // directly consumed by blur_y.
    // However, this introduces significant redundant computation since each
    // blur_x value is recomputed 3 times, once for each blur_y computation
    // that needs it

    blur_y.compute_root();
    blur_x.compute_inline();

    // inline is the default schedule, however. This makes it easy to express
    // long expressions as chains of Funcs without paying a performance price
    // in general, inline is good when the dependency is a single pixel
    // (no branching factor that would introduce redundant computation)
    cout << "\nSchedule 2, INLINE:\t";
    float t = profile(blur_y, width - 2, height - 2);
    cout << "Speedup compared to root: " << (refTime / t) << endl;

    // In effect, this schedule turned a separable blur into the brute force
    // 2D blur
    // The compiler would also  merge the various divisions by 3
    // into a single division by 9, or better a multiplication by the
    // reciprocal. The reciprocal probably would stay in register, making
    // everything blazingly fast. Compilers can be pretty smart.
    blur_y.compile_to_lowered_stmt("blur_schedule2.html", {}, Halide::HTML);
  }

  //////// SCHEDULE 3 : TILING and INTERLEAVING ////////
  {
    Func blur_x("blur_x"); // declare the horizontal blur function
    Func blur_y("blur_y"); // declare the vertical blur function
    blur_x(x, y) =
        cast<float>(input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3.0f;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

    // This is a good schedule (good locality, limited redundancy) that performs
    // computation in tiles and interleaves the two stages of the pipeline
    // within a tile

    // First schedule the last (output) stage
    // In Halide, the schedule is always driven by the output
    // Earlier stages are scheduled with respect to later schedules
    // That is, we schedule a producer with respect to its consumer(s)

    blur_y.tile(x, y, xo, yo, xi, yi, 256, 32); // compute in tiles of 256x32
    // There is also a shorter version of the tile syntax that reuses the
    // original Vars x, y for the outer tile indices: blur_y.tile(x, y, xi, yi,
    // 256, 32)

    // We now specify when the earlier (producer) stage blur_x gets evaluated.
    // We decide to compute it at the tile granularity of blur_y and use the
    // compute_at method.
    // This means that blur_x will be evaluated in a loop nested inside the
    // "xo" outer tile loop of blur_y
    // note that we do not need to specify yo, xi, yi and they are directly
    // inherited from blur_y"s scheduling
    // More importantly, Halide performs automatic bound inference and enlarges
    // the tiles to make sure that all the inputs needed for a tile of blur_y
    // are available. In this case, it means making the blur_x tile one pixel
    // larger above and below to accomodate the 1x3 vertical stencil of blur_y
    // This is all done under the hood and the programmer doesn"t need to worry
    // about it
    blur_x.compute_at(blur_y, xo);

    // This schedule achieves better locality than root but with a lower
    // redundancy than inline. It still has some redundancy because of the
    // enlargement at tile boundaries

    cout << "\nSchedule 3: TILING\t";
    float t = profile(blur_y, width - 2, height - 2);
    cout << "Speedup compared to root: " << (refTime / t) << endl;
    blur_y.compile_to_lowered_stmt("blur_schedule3.html", {}, Halide::HTML);
  }

  //////// SCHEDULE 4 : TILING, INTERLEAVING, and PARALLELISM ////////
  {
    Func blur_x("blur_x"); // declare the horizontal blur function
    Func blur_y("blur_y"); // declare the vertical blur function
    blur_x(x, y) =
        cast<float>(input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3.0f;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3.0f;

    // This is a high-performance schedule that adds multicore and SIMD
    // parallelism to the tiled and interleaved schedule above.

    // First schedule the last (output) stage
    // We specify computation in tiles of 256x32
    blur_y.tile(x, y, xo, yo, xi, yi, 256, 32);
    // We then parallelize the for loop corresponding to the yo tile index
    // Halide will generate multithreaded code and runtime able to take
    // advantage of multicore processors.

    blur_y.parallel(yo);
    // We then specify that we want to use the SIMD vector units
    // (Single Instruction Multiple Data) and compute 8 pixels at once
    // Only try to vectorize the innermost loops.
    // There is no guarantee that the compiler will successfully achieve
    // vectorization For example, if you specify a width larger than what your
    // processor can achieve, it won"t work
    blur_y.vectorize(xi, 8);

    // the above three scheduling instructions can be piped into a more compact
    // version: blur_y.tile(x, y, xo, yo, xi, yi, 256,
    // 32).parallel(yo).vectorize(xi, 8) or with nicer formatting:
    // blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)
    //       .parallel(yo)
    //       .vectorize(xi, 8)

    // We now specify when the earlier (producer) stage blur_x gets evaluated.
    // We decide to compute it at the tile granularity of blur_y and use the
    // compute_at method.
    // This means that blur_x will be evaluated in a loop nested inside the
    // "xo" outer tile loop of blur_y
    // since xo is nested inside blur_y"s yo and since yo is evaluated in
    // parallel, then blur_x will also be evaluated in parallel Again, we don"t
    // need to worry about bound expansion
    blur_x.compute_at(blur_y, xo);

    // We then specify that blur_x too should be vectorized
    // Unlike the parallelism that we inherited from blur_y"s yo loop,
    // vectorization needs to be specified again because its loop nest is lower
    // than the "compute_at" loop xo, whereas yo was above xo. Note that blur_x
    // gets vectorized at x whereas blur_y"s vectorize was called with xi This
    // is because blur_x does not have a notion of x_i. Its tiling piggybacked
    // on that of blur_y, and as far as blur_x is concerned, it just gets called
    // for a tile and has a single granularity of x and y for this tile
    // (although of course vectorize then adds a second.
    blur_x.vectorize(x, 8);

    // This schedule achieves the same excellent locality  and low redundancy
    // as the above tiling and fusion. In addition, it leverages high
    // parallelism.

    cout << "\nSchedule 4: TILE & PARALLEL\t";
    float t = profile(blur_y, width - 2, height - 2);

    blur_y.compile_to_lowered_stmt("blur_schedule4.html", {}, Halide::HTML);

    cout << "Speedup compared to root: " << (refTime / t) << endl;
  }

  cout << "\nSuccess!" << endl;

  // Functional
  {

    Func input("input_fn");
    input(x, y, c) = cast<float>(im(x, y, c));

    // This time, we'll wrap the input in a Func that prevents
    // reading out of bounds:
    Func clamped("clamped");

    // Define an expression that clamps x to lie within the
    // range [0, input.width()-1].
    Expr clamped_x = clamp(x, 0, width - 1);
    // clamp(x, a, b) is equivalent to max(min(x, b), a).

    // Similarly clamp y.
    Expr clamped_y = clamp(y, 0, height - 1);
    // Load from input at the clamped coordinates. This means that
    // no matter how we evaluated the Func 'clamped', we'll never
    // read out of bounds on the input. This is a clamp-to-edge
    // style boundary condition, and is the simplest boundary
    // condition to express in Halide.
    clamped(x, y, c) = input(clamped_x, clamped_y, c);

    Func blur_x("blur_x");
    blur_x(x, y, c) =
        (clamped(x - 2, y, c) + clamped(x - 1, y, c) + clamped(x, y, c) +
         clamped(x + 1, y, c) + clamped(x + 2, y, c)) /
        5.0f;

    // Blur it vertically:
    Func blur_y("blur_y");
    blur_y(x, y, c) =
        (blur_x(x, y - 1, c) + blur_x(x, y - 2, c) + blur_x(x, y, c) +
         blur_x(x, y + 1, c) + blur_x(x, y + 2, c)) /
        5.0f;

    // Convert back to 8-bit.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

    Buffer<uint8_t> result = output.realize(im.width(), im.height(), 3);
    // Save the result. It should look like a slightly blurry
    // parrot, but this time it will be the same size as the
    // input.
    save_image(result, argv[2]); // Benchmark the pipeline.

    blur_y.tile(x, y, xo, yo, xi, yi, 256, 32);
    // We then parallelize the for loop corresponding to the yo tile index
    // Halide will generate multithreaded code and runtime able to take
    // advantage of multicore processors.

    blur_y.parallel(yo);
    // We then specify that we want to use the SIMD vector units
    // (Single Instruction Multiple Data) and compute 8 pixels at once
    // Only try to vectorize the innermost loops.
    // There is no guarantee that the compiler will successfully achieve
    // vectorization For example, if you specify a width larger than what your
    // processor can achieve, it won"t work
    blur_y.vectorize(xi, 8);

    // the above three scheduling instructions can be piped into a more compact
    // version: blur_y.tile(x, y, xo, yo, xi, yi, 256,
    // 32).parallel(yo).vectorize(xi, 8) or with nicer formatting:
    // blur_y.tile(x, y, xo, yo, xi, yi, 256, 32)
    //       .parallel(yo)
    //       .vectorize(xi, 8)

    // We now specify when the earlier (producer) stage blur_x gets evaluated.
    // We decide to compute it at the tile granularity of blur_y and use the
    // compute_at method.
    // This means that blur_x will be evaluated in a loop nested inside the
    // "xo" outer tile loop of blur_y
    // since xo is nested inside blur_y"s yo and since yo is evaluated in
    // parallel, then blur_x will also be evaluated in parallel Again, we don"t
    // need to worry about bound expansion
    blur_x.compute_at(blur_y, xo);

    // We then specify that blur_x too should be vectorized
    // Unlike the parallelism that we inherited from blur_y"s yo loop,
    // vectorization needs to be specified again because its loop nest is lower
    // than the "compute_at" loop xo, whereas yo was above xo. Note that blur_x
    // gets vectorized at x whereas blur_y"s vectorize was called with xi This
    // is because blur_x does not have a notion of x_i. Its tiling piggybacked
    // on that of blur_y, and as far as blur_x is concerned, it just gets called
    // for a tile and has a single granularity of x and y for this tile
    // (although of course vectorize then adds a second.
    blur_x.vectorize(x, 8);
  }

  return EXIT_SUCCESS;
}

////////////////////////// EXERCISES //////////////////////////////////////

// Report the runtime and throughout for the above 4 schedules in README.

// Write the equivalent C++ code for the following Halide schedules:
// for blur_x and blur_y defined in the same way
// You can assume that the image is an integer multiple of tile sizes when
// convenient.

// schedule 5:
// blur_y.compute_root()
// blur_x.compute_at(blur_y, x)

// schedule 6:
// blur_y.tile(x, y, xo, yo, xi, yi, 2, 2)
// blur_x.compute_at(blur_y, yo)

// schedule 7
// blur_y.split(x, xo, xi, 2)
// blur_x.compute_at(blur_y, y)
