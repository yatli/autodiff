#pragma once
#include "nn.hpp"

template<typename T>
struct cnn_t : public nn_t<T> {
  using mat = MatrixXtvar<T>;
  using vec = VectorXtvar<T>;

  int nchannel;
  int width;
  int height;
  int nclass;
  std::vector<ndarray_t<T>> W1, W2, W3;
  ndarray_t<T> b1, b2, b3;
  mat Wf1, Wf2;

  cnn_t(int c, int h, int w, int nclass)
    : nchannel(c), width(w), height(h), nclass(nclass),
      b1(32, h, w, 0.05), b2(64, h/2, w/2, 0.05), b3(64, h/2, w/2, 0.05) {
    W1.reserve(32);
    W2.reserve(64);
    W3.reserve(64);
    for(int i=0;i<32;++i) { W1.emplace_back(c, 3, 3, 0.05); nn_t<T>::register_params(W1[i].v); }
    for(int i=0;i<64;++i) { W2.emplace_back(32, 3, 3, 0.05); nn_t<T>::register_params(W2[i].v); }
    for(int i=0;i<64;++i) { W3.emplace_back(64, 3, 3, 0.05); nn_t<T>::register_params(W3[i].v); }
    nn_t<T>::register_params(b1.v);
    nn_t<T>::register_params(b2.v);
    nn_t<T>::register_params(b3.v);
    Wf1 = mat::Random(512, 64 * h / 4 * w / 4 +1) * 0.05;
    Wf2 = mat::Random(nclass, 512 + 1) * 0.05;
    nn_t<T>::register_params(Wf1);
    nn_t<T>::register_params(Wf2);
  }

  virtual vec forward(const vec& x) {
    ndarray_t<T> x1(x, nchannel, height, width);
    auto x2 = conv2d_layer(x1, W1, b1, act_relu);
    auto x3 = maxpooling_2d(x2, 2, 2);
    dropout(x3.v, 0.25);
    auto x4 = conv2d_layer(x3, W2, b2, act_relu);
    auto x5 = conv2d_layer(x4, W3, b3, act_relu);
    auto x6 = maxpooling_2d(x5, 2, 2);
    dropout(x6.v, 0.25);
    auto x7 = withb(x6.v);
    auto x8 = fc_layer(x7, Wf1, act_relu);
    dropout(x8, 0.5);
    auto x9 = withb(x8);
    auto x10 = fc_layer(x9, Wf2, act_softmax);
    return x10;
  }
};
