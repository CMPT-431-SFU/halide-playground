#include "halide_benchmark.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "stereo_pipe.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: ./run left0224.png left-remap.png right0224.png "
               "right-remap.png\n");
        return 0;
    }

    Buffer<uint8_t> left = load_image(argv[1]);
    Buffer<uint8_t> left_remap = load_image(argv[2]);
    Buffer<uint8_t> right = load_image(argv[3]);
    Buffer<uint8_t> right_remap = load_image(argv[4]);

    Buffer<uint8_t> out(left.width(), left.height());

    printf("start.\n");

    stereo_pipe(right, left, right_remap, left_remap, out);
    save_image(out, "out.png");

    printf("finish running native code\n");
}
