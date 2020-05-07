#pragma once

#include <iostream>
#include <ctime>
#include <chrono>

#include "qnum.hpp"
#include "flex.hpp"
#include "eigen.hpp"

#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

template<typename T> struct is_qnum {
  static constexpr bool value = false;
};

template<typename T, int E> struct is_qnum<qnum::qspace_number_t<T, E>> {
  static constexpr bool value = true;
};

template<typename T> struct is_flexfloat {
  static constexpr bool value = false;
};

template<int E, int F> struct is_flexfloat<flex::flexfloat<E, F>> {
  static constexpr bool value = true;
};

template<typename T> struct is_std_float {
  static constexpr bool value = false;
};

template<> struct is_std_float<float> {
  static constexpr bool value = true;
};

template<> struct is_std_float<double> {
  static constexpr bool value = true;
};

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES_T(autodiff::reverse::Variable<T>, tvar);

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
  autodiff::reverse::Variable<T> sum = 0;
  auto maxi = 0;
  for (auto i = 0; i < n; ++i) {
    if(x[i].expr->val > x[maxi].expr->val) {
      maxi = i;
    }
  }
  for (auto i = 0; i < n; ++i) {
    ret[i] = exp(x[i] - x[maxi]);
    sum += ret[i];
  }
  if (sum == 0) {
    return ret;
  }
  for (auto i = 0; i < n; ++i) {
    ret[i] = ret[i] / sum;
  }
  return ret;
}

template<typename T>
autodiff::reverse::Variable<T> loss_l2(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  autodiff::reverse::Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    sum += (y1[i] - y2[i]) * (y1[i] - y2[i]);
  }
  return sum;
}

template<typename T>
autodiff::reverse::Variable<T> loss_mse(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  autodiff::reverse::Variable<T> norm = T(1.0 / n);
  autodiff::reverse::Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    auto diff = y1[i] - y2[i];
    sum += diff * diff * norm;
  }
  return sum;
}

template<typename T>
autodiff::reverse::Variable<T> loss_crossent(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  autodiff::reverse::Variable<T> sum = 0;
  for (auto i = 0; i < n; ++i) {
    if (y1[i].expr->val == 1) {
      sum += log(y2[i]);
    }
  }
  return -sum;
}

template<typename T>
autodiff::reverse::Variable<T> loss_abs(const VectorXtvar<T>& y1, const VectorXtvar<T>& y2) {
  const auto n = y1.size();
  autodiff::reverse::Variable<T> norm = T(1.0 / n);
  autodiff::reverse::Variable<T> sum = 0;
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

