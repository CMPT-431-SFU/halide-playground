#include "Halide.h"
#include "halide_trace_config.h"
#include <stdint.h>

namespace
{

using std::vector;
using namespace Halide;
using namespace Halide::ConciseCasts;

Var x("x"), y("y"), z("z"), c("c");
Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

class GaussianPipe : public Halide::Generator<GaussianPipe>
{
  public:
    Input<Buffer<uint8_t>> input{"input", 2};
    Output<Buffer<uint8_t>> output{"output"};

    void generate()
    {

        Func in_bounded("in_bounded"), kernel("kernel"), kernel_f("kernel_f"),
            sum_x("sum_x"), sum_y("sum_y"), blur_y("blur_y"), blur_x("blur_x");
        RDom win(0, 2), win2(-4, 9, -4, 9);
        // Define a 9x9 Gaussian Blur with a
        // repeat-edge boundary condition.
        float sigma = 1.5f;

        kernel_f(x) = exp(-x * x / (2 * sigma * sigma)) / (sqrtf(2 * M_PI) * sigma);

        // Normalize and convert to 8bit fixed point.
        // Kernel values will inlined into  the blurring kernel as constant
        kernel(x) = cast<uint8_t>(kernel_f(x) * 255 /
                                  (kernel_f(0) + kernel_f(1) * 2 + kernel_f(2) * 2 +
                                   kernel_f(3) * 2 + kernel_f(4) * 2));

        // in_bounded = BoundaryConditions::repeat_edge(in);
        in_bounded(x, y) = input(x + 4, y + 4);

        // 2D filter: direct map
        sum_x(x, y) += cast<uint32_t>(in_bounded(x + win2.x, y + win2.y)) *
                       kernel(win2.x) * kernel(win2.y);
        blur_x(x, y) = cast<uint8_t>(sum_x(x, y) >> 16);

        sum_x.update(0).unroll(win2.x).unroll(win2.y);

        output(x, y) = blur_x(x, y);
        output.tile(x, y, xo, yo, xi, yi, 256, 64)
            .vectorize(xi, 8)
            .fuse(xo, yo, xo)
            .parallel(xo);
    }
}; // namespace
} // namespace
HALIDE_REGISTER_GENERATOR(GaussianPipe, gaussian_pipe)