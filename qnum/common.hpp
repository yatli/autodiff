#pragma once

#include <iostream>
#include <ctime>
#include <chrono>
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
VectorXvar<T> withb(const VectorXvar<T>& x) {

  const auto n = x.size();
  VectorXvar<T> ret(n+1);
  ret[0] = T(1.0);
  for (auto i = 0; i < n; ++i) {
    ret[i+1] = x[i];
  }
  return ret;
}


template<typename T>
VectorXvar<T> act_sigmoid(const VectorXvar<T>& x) {
  const auto n = x.size();
  VectorXvar<T> ret(n);
  for (auto i = 0; i < n; ++i)
  {
    ret[i] = autodiff::sigmoid(x[i]);
  }
  return ret;
}

template<typename T>
VectorXvar<T> act_relu(const VectorXvar<T>& x) {
  const auto n = x.size();
  VectorXvar<T> ret(n);
  for (auto i = 0; i < n; ++i)
  {
    ret[i] = autodiff::relu(x[i]);
  }
  return ret;
}

template<typename T>
VectorXvar<T> act_softmax(const VectorXvar<T>& x) {

  const auto n = x.size();
  VectorXvar<T> ret(n);
  var<T> sum = 0;
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
var<T> loss_l2(const VectorXvar<T>& y1, const VectorXvar<T>& y2) {
  const auto n = y1.size();
  var<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum += (y1[i] - y2[i]) * (y1[i] - y2[i]);
  }
  return sum;
}

template<typename T>
var<T> loss_mse(const VectorXvar<T>& y1, const VectorXvar<T>& y2) {
  const auto n = y1.size();
  var<T> norm = T(1.0 / n);
  var<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    auto diff = y1[i] - y2[i];
    sum += diff * diff * norm;
  }
  return sum;
}

template<typename T>
var<T> loss_crossent(const VectorXvar<T>& y1, const VectorXvar<T>& y2) {
  const auto n = y1.size();
  var<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum -= y1[i] * log(y2[i]);
  }
  return sum;
}

template<typename T>
var<T> loss_abs(const VectorXvar<T>& y1, const VectorXvar<T>& y2) {
  const auto n = y1.size();
  var<T> norm = T(1.0 / n);
  var<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum += abs(y1[i] - y2[i]) * norm;
  }
  return sum;
}

template<typename T>
VectorXvar<T> fc_layer(const VectorXvar<T>& x, const MatrixXvar<T>& W, VectorXvar<T>(f)(const VectorXvar<T>&))
{
  VectorXvar<T> v = W * x;
  return f(v);
}

