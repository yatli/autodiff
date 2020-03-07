#include <iostream>
#include <ctime>
using namespace std;

#include <Eigen/Core>
using namespace Eigen;

#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
using namespace autodiff;

#include "qnum.hpp"
#include "eigen.hpp"
using namespace qnum;

#define debug_dump(x) \
  do { std::cout << #x << " = " << x << std::endl; } while (false)

template<typename T>
VectorXvar<T> forward_pass(const VectorXvar<T>& x, const MatrixXvar<T>& W, const VectorXvar<T>& b)
{
  return W * x + b;
}

template<typename T, typename U>
T silly(const U x) {
  return T(x);
}

void sanity_check() {
  debug_dump(std::numeric_limits<int32_t>::digits);
  debug_dump(qnum::qnum32_t::T_max());
  debug_dump(qnum::qnum32_t::ext_bits());
  debug_dump(qnum::qnum32_t::ext_max());
  debug_dump(qnum::qnum32_t::frac_bits());
  debug_dump(qnum::qnum32_t::frac_max());

  debug_dump(qnum32_t(0.1));
  debug_dump(qnum32_t(0.1).prev());
  debug_dump(qnum32_t(0.1).next());
  debug_dump(qnum32_t(0.01));
  debug_dump(qnum32_t(0.001));
  debug_dump(qnum32_t(0.0001));
  debug_dump(qnum32_t(0.00001));
}

void vector_check() {
  VectorXvar<qnum16_t> x(5);
  x << qnum16_t(0.1), qnum16_t(0.12), qnum16_t(0.14), qnum16_t(0.16), qnum16_t(0.18);

  MatrixXvar<qnum16_t> W = MatrixXvar<qnum16_t>::Random(5, 5);

  debug_dump(x);
  debug_dump(W);
  debug_dump(W * x);
}

void autodiff_check() {
  int vec_size = 128;
  VectorXvar<qnum16_t> x = VectorXvar<qnum16_t>::Random(vec_size);
  MatrixXvar<qnum16_t> W = MatrixXvar<qnum16_t>::Random(vec_size, vec_size);
  VectorXvar<qnum16_t> b = VectorXvar<qnum16_t>::Random(vec_size);

  ////try autodiff
  VectorXvar<qnum16_t> y = forward_pass(x, W, b);
  var<qnum16_t> sum = y.sum();
  auto dsum_dx = gradient(sum, x);

  std::cout << dsum_dx << std::endl;
}

#define run(x) \
  do { \
    std::cout << "---------- run " << #x << " -----------" << std::endl; \
    x(); \
  } while (false)

int main(int argc, char** argv) {
  srand(time(nullptr));

  run(sanity_check);
  run(vector_check);
  run(autodiff_check);

  return 0;
}
