// compute: \vec{x}^{\top} \vec{y}
export uniform float dotf(uniform float const x[],
                          uniform float const y[],
                          uniform int n) {
  float retval = 0;
  foreach(i = 0 ... n) {
    retval += x[i] * y[i];
  }
  return reduce_add(retval);
}

// compute: \vec{x}^{\top} \vec{y}
export uniform float dot(uniform double const x[],
                         uniform double const y[],
                         uniform int n) {
  double retval = 0;
  foreach(i = 0 ... n) {
    retval += x[i] * y[i];
  }
  return reduce_add(retval);
}

// compute: \vec{y} += \vec{x} * a
export void axpyf(uniform float const a,
                  uniform float const x[],
                  uniform float       y[],
                  uniform int n) {
  foreach(i = 0 ... n) {
    y[i] += x[i] * a;
  }
}

// compute: \vec{y} += \vec{x} * a
export void axpy(uniform double const a,
                 uniform double const x[],
                 uniform double       y[],
                 uniform int n) {
  foreach(i = 0 ... n) {
    y[i] += x[i] * a;
  }
}

// compute: C = A B^{\top}
export void gemmf(uniform float const a[],
                  uniform float const b[],
                  uniform float       c[],
                  uniform int m,
                  uniform int n,
                  uniform int r) {
  for (uniform int i = 0; i < n; ++i) {
    for (uniform int j = 0; j < m; ++j) {
      c[i * m + j] = dotf(a + i * r, b + j * r, r);
    }
  }
}

// compute: C = A B^{\top}
export void gemm(uniform double const a[],
                 uniform double const b[],
                 uniform double       c[],
                 uniform int m,
                 uniform int n,
                 uniform int r) {
  for (uniform int i = 0; i < n; ++i) {
    for (uniform int j = 0; j < m; ++j) {
      c[i * m + j] = dot(a + i * r, b + j * r, r);
    }
  }
}
