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
  }
  time = omp_get_wtime() - time;
  std::cout << time << ' ' << (double(B) * (S - R) * (S - R) * K * R * R * C) / (time * 1e+9) << " GFLOPS\n";
}
