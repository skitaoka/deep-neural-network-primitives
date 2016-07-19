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

  std::vector<float> const  x(B * S * S * C);
  std::vector<float> const dy(B * S * S * K);
  std::vector<float>       dw(K * R * R * C); // backprop ‘O‚É 0 ƒNƒŠƒA‚µ‚Ä‚¨‚­

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
            float const a = dy[((b * S + (j + R / 2)) * S + (i + R / 2)) * K + k];
            for (int jj = 0; jj < R; ++jj) {
              ispc::axpyf(a,
                          x .data() + ((b * S + (jj + j)) * S + i) * C,
                          dw.data() + ( k * R +  jj     ) * R      * C, R * C);
            }
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
            for (int jj = 0; jj < R; ++jj) {
              for (int ii = 0; ii < R; ++ii) {
                for (int c = 0; c < C; ++c) {
                  dw[((k * R + jj) * R + ii) * C + c]
                    += x[((b * S + (j + jj )) * S + (i + ii )) * C + c]
                    * dy[((b * S + (j + R/2)) * S + (i + R/2)) * K + k];
                }
              }
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
