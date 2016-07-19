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

  std::vector<float> const x(B * S * S * C);
  std::vector<float>       y(B * S * S * K); // ˆ—‘O‚É 0 ƒNƒŠƒA‚µ‚Ä‚¨‚­
  std::vector<float> const w(K * R * R * C);

  double time = omp_get_wtime();
  {
#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int j = 0; j <= S - R; ++j) {
        for (int i = 0; i <= S - R; ++i) {
          for (int k = 0; k < K; ++k) {
            float retval = {};
            for (int jj = 0; jj < R; ++jj) {
              retval += ispc::dotf(w.data() + ( k * R + jj    ) * R      * C,
                                   x.data() + ((b * S + jj + j) * S + i) * C, C * R);
            }
            y[((b * S + (j + R / 2)) * S + (i + R / 2)) * K + k] += retval;
          }
        }
      }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int j = 0; j <= S - R; ++j) {
        for (int i = 0; i <= S - R; ++i) {
          for (int k = 0; k < K; ++k) {
            float retval = {};
            for (int jj = 0; jj < R; ++jj) {
              retval += ispc::dotf(w.data() + (k * R + jj) * R * C,
                                   x.data() + ((b * S + (jj + j)) * S + i) * C, C * R);
            }
            y[((b * S + (j + R / 2)) * S + (i + R / 2)) * K + k] += retval;
          }
        }
      }
    }
#endif
  }
  time = omp_get_wtime() - time;
  std::cout << time << ' ' << (double(B) * (S - R) * (S - R) * K * R * R * C) / (time * 1e+9) << " GFLOPS\n";
}
