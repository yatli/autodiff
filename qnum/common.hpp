#pragma once

#include <iostream>
#include <ctime>
#include <chrono>
using namespace std;

#include "qnum.hpp"
#include "eigen.hpp"
using namespace qnum;
using namespace Eigen;

#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
using namespace autodiff;


AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES_T(autodiff::Variable<T>, tvar);

#define debug_dump(x) \
  do { std::cout << #x << " = " << x << std::endl; } while (false)

template<typename T>
VectorXtvar<T> withb(const VectorXtvar<T>& x) {

  const auto n = x.size();
  VectorXtvar<T> ret(n+1);
  ret[0] = T(1.0);
  for (auto i = 0; i < n; ++i) {
    ret[i+1] = x[i];
  }
  return ret;
}


template<typename T>
VectorXtvar<T> act_sigmoid(const VectorXtvar<T>& x) {
  const auto n = x.size();
  VectorXtvar<T> ret(n);
  for (auto i = 0; i < n; ++i)
  {
    ret[i] = autodiff::reverse::sigmoid(x[i]);
  }
  return ret;
}

template<typename T>
VectorXtvar<T> act_relu(const VectorXtvar<T>& x) {
  const auto n = x.size();
  VectorXtvar<T> ret(n);
  for (auto i = 0; i < n; ++i)
  {
    ret[i] = autodiff::reverse::relu(x[i]);
  }
  return ret;
}

template<typename T>
VectorXtvar<T> act_softmax(const VectorXtvar<T>& x) {

  const auto n = x.size();
  VectorXtvar<T> ret(n);
  Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    ret[i] = exp(x[i]);
    sum += ret[i];
  }
  for (auto i = 0; i < n; ++i) {
    ret[i] = ret[i] / sum;
  }
  return ret;
}

template<typename T>
Variable<T> loss_l2(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum += (y1[i] - y2[i]) * (y1[i] - y2[i]);
  }
  return sum;
}

template<typename T>
Variable<T> loss_mse(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  Variable<T> norm = T(1.0 / n);
  Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    auto diff = y1[i] - y2[i];
    sum += diff * diff * norm;
  }
  return sum;
}

template<typename T>
Variable<T> loss_crossent(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    if (y1[i].expr->val == 1) {
      sum += log(y2[i]);
    }
  }
  return -sum;
}

template<typename T>
Variable<T> loss_abs(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  Variable<T> norm = T(1.0 / n);
  Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum += abs(y1[i] - y2[i]) * norm;
  }
  return sum;
}

template<typename T>
VectorXtvar<T> fc_layer(const VectorXtvar<T>& x, const MatrixXtvar<T>& W, VectorXtvar<T>(f)(const VectorXtvar<T>&))
{
  VectorXtvar<T> v = W * x;
  return f(v);
}

template<typename T>
int argmax(const VectorXtvar<T>& x) {
  const int n = x.size();
  int ret = 0;
  double maxval = x[0].expr->val;
  for (auto i = 1; i < n; ++i) {
    auto xi = static_cast<double>(x[i].expr->val);
    if (xi >= maxval) {
      maxval = xi;
      ret = i;
    }
  }
  return ret;
}

