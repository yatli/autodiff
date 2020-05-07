#include "common.hpp"
using namespace std;

void growth_mul_check() {
  qnum::qnum16_t<> v = 0.1;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v *= 2;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
}

void growth_add_check() {
  qnum::qnum16_t<> v = 0.1;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);
  v += 1.25;
  debug_dump(v);
  debug_dump(v.growth);
  debug_dump(v.val);


}

void q16_consts_check() {
  debug_dump(std::numeric_limits<int16_t>::digits);
  debug_dump(qnum::qnum16_t<>::T_max());
  debug_dump(qnum::qnum16_t<>::ext_bits());
  debug_dump(qnum::qnum16_t<>::joint_bits());
  debug_dump(qnum::qnum16_t<>::frac_bits());
  debug_dump(qnum::qnum16_t<>::ext_max());
  debug_dump(qnum::qnum16_t<>::frac_max());
  debug_dump(qnum::qnum16_t<>::K());
}

void q32_step_check() {
  debug_dump(qnum32_t<>(0.1));
  debug_dump(qnum32_t<>(0.1).prev());
  debug_dump(qnum32_t<>(0.1).next());
  debug_dump(qnum32_t<>(0.01));
  debug_dump(qnum32_t<>(0.001));
  debug_dump(qnum32_t<>(0.0001));
  debug_dump(qnum32_t<>(0.00001));
}

void qnum_iter_check() {
  for (qnum8_t<> i = -1.0; i < 0.99; i = i.next()) {
    debug_dump(i);
  }

  for (qnum16_t<> i = -1.0; i < 0.99; i = i.next()) {
    debug_dump(i);
  }
}

void q32_add_check() {

  debug_dump(qnum32_t<>(0.12) + qnum32_t<>(0.3456));
  debug_dump(qnum32_t<>(-0.151248) * qnum32_t<>(2.4));

}

void flex_check() {
  flex::float16_t a = 3.14;
  flex::float16_t b = 2.0;
  debug_dump(a*b);
}

void flex_autodiff_check() {
  VectorXtvar<flex::float16_t> x(5);
  x << 0.1, 0.2, 0.3, 0.4, 0.5 ;
  MatrixXtvar<flex::float16_t> W = MatrixXtvar<flex::float16_t>::Random(5, 5);

  debug_dump(x);
  debug_dump(W);
  debug_dump(W * x);
}

void vector_check() {
  VectorXtvar<qnum16_t<>> x(5);
  x << qnum16_t<>(0.1), qnum16_t<>(0.12), qnum16_t<>(0.14), qnum16_t<>(0.16), qnum16_t<>(0.18);

  MatrixXtvar<qnum16_t<>> W = MatrixXtvar<qnum16_t<>>::Random(5, 5);

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

  run(flex_check);
  run(flex_autodiff_check);
  //run(q32_add_check);
  //run(growth_add_check);
  //run(growth_mul_check);
  //run(vector_check);
  //run(autodiff_check<qnum64_t<>>);
  //run(autodiff_check<qnum32_t<>>);
  //run(autodiff_check<qnum16_t<>>);
  //run(autodiff_check<double>);
  //run(autodiff_check<float>);

  return 0;
}
