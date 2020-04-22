#include "common.hpp"

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

  debug_dump(qnum32_t(0.12) + qnum32_t(0.3456));
  debug_dump(qnum32_t(-0.151248) * qnum32_t(2.4));

  //for (qnum8_t i = -1.0; i < 0.99; i = i.next()) {
  //  debug_dump(i);
  //}

  //for (qnum16_t i = -1.0; i < 0.99; i = i.next()) {
  //  debug_dump(i);
  //}
}

void vector_check() {
  VectorXtvar<qnum16_t> x(5);
  x << qnum16_t(0.1), qnum16_t(0.12), qnum16_t(0.14), qnum16_t(0.16), qnum16_t(0.18);

  MatrixXtvar<qnum16_t> W = MatrixXtvar<qnum16_t>::Random(5, 5);

  debug_dump(x);
  debug_dump(W);
  debug_dump(W * x);
}

template<typename T>
void autodiff_check() {
  int vec_size = 128;
  int vec_size2 = 10;
  VectorXtvar<T> x = VectorXtvar<T>::Random(vec_size);
  MatrixXtvar<T> W1 = MatrixXtvar<T>::Random(vec_size, vec_size) * 0.05;
  MatrixXtvar<T> W2 = MatrixXtvar<T>::Random(vec_size2, vec_size) * 0.05;

  ////try autodiff
  VectorXtvar<T> x1 = fc_layer(x, W1, act_sigmoid);
  VectorXtvar<T> y = fc_layer(x1, W2, act_softmax);
  VectorXtvar<T> y_rand = VectorXtvar<T>::Zero(vec_size2);
  y_rand[0] = T(1.0);

  //auto loss = loss_l2(y, y_rand);
  auto loss = loss_mse(y, y_rand);
  // XXX only for vecs now
  //auto gw2 = gradient(loss, W2);
  //auto gw1 = gradient(loss, W1);
}

#define run(x) \
  do { \
    chrono::high_resolution_clock clock; \
    std::cout << "---------- run " << #x << " -----------" << std::endl; \
    auto t1 = clock.now(); \
    x(); \
    auto t2 = clock.now(); \
    std::cout << "---------- completed " << #x << " ( " << (chrono::duration_cast<chrono::milliseconds>(t2 - t1)).count() << "ms ) -----------" << std::endl; \
  } while (false)

int main(int argc, char** argv) {
  srand(time(nullptr));

  run(sanity_check);
  run(vector_check);
  run(autodiff_check<qnum64_t>);
  run(autodiff_check<qnum32_t>);
  run(autodiff_check<qnum16_t>);
  run(autodiff_check<double>);
  run(autodiff_check<float>);

  return 0;
}
