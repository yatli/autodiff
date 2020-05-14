#pragma once
#include "nn.hpp"
#include "common.hpp"
#include <vector>
#include <cstdlib>

template<typename T>
struct mlp_t : public nn_t<T> {
  using mat = MatrixXtvar<T>;
  using vec = VectorXtvar<T>;
  mat w1, w2;
  int sz_input, sz_hidden, sz_output;

  mlp_t(int ninput, int nhidden, int noutput) :
    sz_input(ninput), sz_hidden(nhidden), sz_output(noutput),
    w1(mat::Random(nhidden, ninput + 1) * 0.05),
    w2(mat::Random(noutput, nhidden + 1) * 0.05) { 

    nn_t<T>::register_params(w1);
    nn_t<T>::register_params(w2);
  }

  virtual vec forward(const vec& x) {
    auto bx = withb(x);
    auto hx = withb(fc_layer(bx, w1, act_relu));
    auto ox = fc_layer(hx, w2, act_identity);
    return ox;
  }

  vec forward_debug(const vec& x) {
    VectorXtvar<T> bx = withb(x);
    debug_dump(bx);
    VectorXtvar<T> p1 = w1 * bx;
    debug_dump(p1);
    VectorXtvar<T> hx = withb(act_relu(p1));
    debug_dump(hx);
    VectorXtvar<T> p2 = w2 * hx;
    debug_dump(p2);
    VectorXtvar<T> ox = act_softmax(p2);
    debug_dump(ox);
    return ox;
  }

};

