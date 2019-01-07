#include "Halide.h"
#include "halide_trace_config.h"
#include <stdint.h>

namespace
{

using std::vector;

using namespace Halide;
using namespace Halide::ConciseCasts;

Var x("x"), y("y"), c("c");
Var xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile_index("ti");
Var xio("xio"), yio("yio"), xv("xv"), yp("yp");

int blockSize = 3;
int Ksize = 3;

float k = 0.04;
float threshold = 100;

class HarrisPipe : public Halide::Generator<HarrisPipe>
{
  public:
    // Parameterized output type, because LLVM PTX (GPU) backend does not
    // currently allow 8-bit computations
    GeneratorParam<Type> result_type{"result_type", UInt(8)};

    Input<Buffer<uint8_t>> input{"input", 2};
    Output<Buffer<uint8_t>> output{"output", 2};

    void generate()
    {
        Func padded("padded");
        Func cim("cim");
        Func grad_x("grad_x"), grad_y("grad_y");
        Func grad_xx("grad_xx"), grad_yy("grad_yy"), grad_xy("grad_xy");
        Func grad_gx("grad_gx"), grad_gy("grad_gy"), grad_gxy("grad_gxy");
        //   Func output("output");
        RDom box(-blockSize / 2, blockSize, -blockSize / 2, blockSize),
            maxWin(-1, 3, -1, 3);

        // padded = BoundaryConditions::repeat_edge(input);
        padded(x, y) = input(x + 3, y + 3);

        // sobel filter
        Func padded16;
        padded16(x, y) = cast<int16_t>(padded(x, y));
        grad_x(x, y) =
            cast<int16_t>(-padded16(x - 1, y - 1) + padded16(x + 1, y - 1) -
                          2 * padded16(x - 1, y) + 2 * padded16(x + 1, y) -
                          padded16(x - 1, y + 1) + padded16(x + 1, y + 1));
        grad_y(x, y) =
            cast<int16_t>(padded16(x - 1, y + 1) - padded16(x - 1, y - 1) +
                          2 * padded16(x, y + 1) - 2 * padded16(x, y - 1) +
                          padded16(x + 1, y + 1) - padded16(x + 1, y - 1));

        grad_xx(x, y) = cast<int32_t>(grad_x(x, y)) * cast<int32_t>(grad_x(x, y));
        grad_yy(x, y) = cast<int32_t>(grad_y(x, y)) * cast<int32_t>(grad_y(x, y));
        grad_xy(x, y) = cast<int32_t>(grad_x(x, y)) * cast<int32_t>(grad_y(x, y));

        // box filter (i.e. windowed sum)
        grad_gx(x, y) += grad_xx(x + box.x, y + box.y);
        grad_gy(x, y) += grad_yy(x + box.x, y + box.y);
        grad_gxy(x, y) += grad_xy(x + box.x, y + box.y);

        // calculate Cim
        int scale = (1 << (Ksize - 1)) * blockSize;
        Expr lgx = cast<float>(grad_gx(x, y) / scale / scale);
        Expr lgy = cast<float>(grad_gy(x, y) / scale / scale);
        Expr lgxy = cast<float>(grad_gxy(x, y) / scale / scale);
        Expr det = lgx * lgy - lgxy * lgxy;
        Expr trace = lgx + lgy;
        cim(x, y) = det - k * trace * trace;

        // Perform non-maximal suppression
        Expr is_max = cim(x, y) > cim(x - 1, y - 1) && cim(x, y) > cim(x, y - 1) &&
                      cim(x, y) > cim(x + 1, y - 1) && cim(x, y) > cim(x - 1, y) &&
                      cim(x, y) > cim(x + 1, y) && cim(x, y) > cim(x - 1, y + 1) &&
                      cim(x, y) > cim(x, y + 1) && cim(x, y) > cim(x + 1, y + 1);
        output(x, y) =
            select(is_max && (cim(x, y) >= threshold), cast<uint8_t>(255), 0);

        /* Schedule */
        output.tile(x, y, xo, yo, xi, yi, 240, 320);
        grad_x.compute_at(output, xo).vectorize(x, 8);
        grad_y.compute_at(output, xo).vectorize(x, 8);
        grad_xx.compute_at(output, xo).vectorize(x, 4);
        grad_yy.compute_at(output, xo).vectorize(x, 4);
        grad_xy.compute_at(output, xo).vectorize(x, 4);
        grad_gx.compute_at(output, xo).vectorize(x, 4);
        grad_gy.compute_at(output, xo).vectorize(x, 4);
        grad_gxy.compute_at(output, xo).vectorize(x, 4);
        cim.compute_at(output, xo).vectorize(x, 4);

        grad_gx.update(0).unroll(box.x).unroll(box.y);
        grad_gy.update(0).unroll(box.x).unroll(box.y);
        grad_gxy.update(0).unroll(box.x).unroll(box.y);

        output.fuse(xo, yo, xo).parallel(xo).vectorize(xi, 4);
    }
};

} // namespace

HALIDE_REGISTER_GENERATOR(HarrisPipe, harris_pipe)
