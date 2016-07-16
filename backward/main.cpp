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
#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y <= S - R; ++y) {
        for (int x = 0; x <= S - R; ++x) {
          for (int c = 0; c < C; ++c) {
            float retval = 0;
            for (int yy = 0; yy < R; ++yy) {
              retval += ispc::dotf(
                filter.data() + (c * R + yy) * R * K,
                output.data() + ((b * S + (yy + y)) * S + x) * K, R * K);
            }
            input[((b * S + (y + R / 2)) * S + (x + R / 2)) * C + c] = retval;
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
          for (int c = 0; c < C; ++c) {
            float retval = 0;
            for (int yy = 0; yy < R; ++yy) {
              for (int xx = 0; xx < R; ++xx) {
                for (int k = 0; k < K; ++k) {
                  retval += filter[((c * R + yy) * R + xx) * K + k]
                    * output[((b * S + (yy + y)) * S + (xx + x)) * K + k];
                }
              }
            }
            input[((b * S + (y + R / 2)) * S + (x + R / 2)) * C + c] = retval;
          }
        }
      }
    }
#endif
  }
  time = omp_get_wtime() - time;
  std::cout << time << ' ' << (double(B) * (S - R) * (S - R) * K * R * R * C) / (time * 1e+9) << " GFLOPS\n";
}
