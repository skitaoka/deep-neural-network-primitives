#include <vector>
#include <array>
#include <iostream>
#include <omp.h>

#include "dot_ispc.h"

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
#if 0
    // 1x1 conv
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
          for (int k = 0; k < K / 2; ++k) {
            output[((b * S + y) * S + x) * K + k]
              = ispc::dot(&filter[k * C],
                          &input[((b * S + y) * S + x) * C], C);
          }
        }
      }
    }
#elif 0
    // 3x3 conv
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y <= S - R; ++y) {
        for (int x = 0; x <= S - R; ++x) {
          for (int k = 0; k < K; ++k) {
            output[((b * S + y + R / 2) * S + x + R / 2) * K + k]
              = ispc::dot(filter.data() + (k * R + 0    ) * R * C,
                          input.data() + ((b * S + 0 + y) * S + x) * C, C * R)
              + ispc::dot(filter.data() + (k * R + 1    ) * R * C,
                          input.data() + ((b * S + 1 + y) * S + x) * C, C * R)
              + ispc::dot(filter.data() + (k * R + 2    ) * R * C,
                          input.data() + ((b * S + 2 + y) * S + x) * C, C * R);
          }
        }
      }
    }
#else
    // RxR conv
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y <= S - R; ++y) {
        for (int x = 0; x <= S - R; ++x) {
          for (int k = 0; k < K; ++k) {
            float retval = {};
            for (int yy = 0; yy < R; ++yy) {
              retval += ispc::dot(filter.data() + (k * R + yy) * R * C,
                                  input.data() + ((b * S + yy + y) * S + x) * C, C * R);
            }
            output[((b * S + y + R / 2) * S + x + R / 2) * K + k] = retval;
          }
        }
      }
    }
#endif
  }
  time = omp_get_wtime() - time;
  std::cout << time;
}