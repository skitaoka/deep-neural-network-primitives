#include <vector>
#include <array>
#include <iostream>
#include <omp.h>

#include "../primitives/lib_ispc.h"

int main() {
  constexpr int B = 192; // batch size
  constexpr int C = 192; // the number of input feature maps
  constexpr int K = 192; // the number of output feature maps
  constexpr int S = 21; // size of input feature map
  constexpr int R = 3; // size of filter

  std::vector<float> input(B * S * S * C);
  std::vector<float> output(B * S * S * K);
  std::vector<float> filter(K * R * R * C);

  double time = omp_get_wtime();
  {
    // RxR conv
#if 1

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        for (int yy = 0; yy < R; ++yy) {
          for (int xx = 0; xx < R; ++xx) {
            float retval = 0.0f;
            for (int y = 0; y <= S - R; ++y) {
              retval += ispc::dotf(input.data() + ((c * S + yy + y) * S + xx) * B,
                                   output.data() + ((k * S + R / 2 + y) * S + R / 2) * B, B * (S - R + 1));
            }
            filter[((k * R + yy) * R + xx) * C + c] = retval;
          }
        }
      }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y <= S - R; ++y) {
        for (int x = 0; x <= S - R; ++x) {
          for (int k = 0; k < K; ++k) {
            float const a = output[((b * S + (y + R / 2)) * S + (x + R / 2)) * K + k];
            for (int yy = 0; yy < R; ++yy) {
              ispc::axpyf(a,
                          input.data() + ((b * S + (y + yy)) * S + x) * C,
                          filter.data() + (k * R + yy) * R * C, R * C);
            }
          }
        }
      }
    }
#endif
  }
  time = omp_get_wtime() - time;
  std::cout << time << ' ' << (double(B) * (S - R) * (S - R) * K * R * R * C) / (time * 1e+9) << " GFLOPS\n";
}
