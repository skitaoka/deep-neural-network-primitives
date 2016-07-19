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

  std::vector<float>       dx(B * S * S * C); // backprop 前に 0 クリアしておく
  std::vector<float> const dy(B * S * S * K);
  std::vector<float> const  w(K * R * R * C); // backprop 前に k と c を転置しておく

  double time = omp_get_wtime();
  {
#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int j = 0; j <= S - R; ++j) {
        for (int i = 0; i <= S - R; ++i) {
          for (int c = 0; c < C; ++c) {
            float retval = {};
            for (int jj = 0; jj < R; ++jj) {
              retval += ispc::dotf(
                 w.data() + ( c * R +  jj     ) * R      * K,
                dy.data() + ((b * S + (jj + j)) * S + i) * K, R * K);
            }
            dx[((b * S + (j + R / 2)) * S + (i + R / 2)) * C + c] += retval;
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
          for (int c = 0; c < C; ++c) {
            float retval = 0;
            for (int jj = 0; jj < R; ++jj) {
              for (int ii = 0; ii < R; ++ii) {
                for (int k = 0; k < K; ++k) {
                  retval += w[((c * R +  jj     ) * R +  ii     ) * K + k]
                         * dy[((b * S + (jj + j)) * S + (ii + i)) * K + k];
                }
              }
            }
            dx[((b * S + (j + R / 2)) * S + (i + R / 2)) * C + c] += retval;
          }
        }
      }
    }
#endif
  }
  time = omp_get_wtime() - time;
  std::cout << time << ' ' << (double(B) * (S - R) * (S - R) * K * R * R * C) / (time * 1e+9) << " GFLOPS\n";
}
