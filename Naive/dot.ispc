export uniform float dot(uniform float const a[], uniform float const b[], uniform int n) {
  float retval = 0.0f;
  foreach(i = 0 ... n) {
    retval += a[i] * b[i];
  }
  return reduce_add(retval);
}
