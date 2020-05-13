#pragma once

#if !NDEBUG
#define PARTIAL_BUILD
#endif

#include <iostream>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <random>

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

/// xavier initialization
std::normal_distribution<double> glorot_normal(int nin, int nout) {
  double dev = std::sqrt(2.0 / (nin + nout));
  return std::normal_distribution<double>(0.0, dev);
}

template<typename T>
MatrixXtvar<T> weight_init(int nin, int nout, bool bias) {
  if (bias) {
    ++nin;
  }
  MatrixXtvar<T> ret;
  ret.resize(nout, nin);
  std::default_random_engine rng;
  auto dist = glorot_normal(nin, nout);
  for(int r = 0; r < nout; ++r) {
    for(int c = 0; c < nin; ++c) {
      if (c == 0 && bias) {
        ret(r, c) = T(0.0);
      } else {
        ret(r, c) = dist(rng);
      }
    }
  }
  return ret;
}

template<typename T>
struct ndarray_t {
  VectorXtvar<T> v;
  int c; int h; int w;

  ndarray_t(VectorXtvar<T> v, int c, int h, int w)
    : v(v), c(c), h(h), w(w) {
    assert(v.size() == c * h * w);
  }

  ndarray_t(int nin, int h, int w)
    : c(nin), h(h), w(w) {
    v.resize(nin*h*w);
  }

  /// with xavier initialization.
  ndarray_t(int nin, int h, int w, int nout)
    : c(nin), h(h), w(w) {
    v.resize(nin*h*w);
    std::default_random_engine rng;
    auto dist = glorot_normal(nin * h * w, nout * h * w);
    for(int i=0; i < v.size(); ++i) {
      v[i] = dist(rng);
    }
  }

  /// with uniform initialization.
  ndarray_t(int c, int h, int w, double range): c(c), h(h), w(w) {
    v = VectorXtvar<T>::Random(c*h*w) * range;
  }

  autodiff::reverse::Variable<T>& operator () (int ch, int y, int x) {
    return v[ch * h * w + y * w + x];
  }

};

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
  autodiff::reverse::Variable<T> sum = autodiff::reverse::constant(T(0.0));
  T maxv = x[0].expr->val;
  for (auto i = 1; i < n; ++i) {
    if(x[i].expr->val > maxv) {
      maxv = x[i].expr->val;
    }
  }
  //if (maxv >= 10) {
    //std::cout << "big maxv in softmax: " << maxv << std::endl;
  //}
  auto cmaxv = autodiff::reverse::constant(maxv);
  for (auto i = 0; i < n; ++i) {
    ret[i] = exp(x[i] - cmaxv);
    //if (ret[i].expr->val == 0) {
      //std::cout << "zero in softmax" << std::endl;
    //}
    sum += ret[i];
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
  for (auto i = 0; i < n; ++i) {
    if (y1[i].expr->val == 1) {
      return -log(y2[i]);
    }
  }
  return 0;
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
VectorXtvar<T> fc_layer(const VectorXtvar<T>& x, const MatrixXtvar<T>& W, VectorXtvar<T>(f)(const VectorXtvar<T>&)) {
  VectorXtvar<T> v;
  v.resize(W.rows());

  for(int i = 0; i < W.rows(); ++i) {
    std::vector<autodiff::reverse::Variable<T>> xs;
    xs.reserve(W.cols());
    for(int j = 0; j < W.cols(); ++j) {
      xs.push_back(x[j] * W(i, j));
    }
    v[i] = autodiff::reverse::sum(xs);
  }

  return f(v);
}

template<typename T>
autodiff::reverse::Variable<T> conv2d_at(ndarray_t<T> &convin, ndarray_t<T> &kernel, int y, int x) {
  assert(kernel.c == convin.c);
  y -= kernel.h / 2;
  x -= kernel.w / 2;
  std::vector<autodiff::reverse::Variable<T>> xs;
  xs.reserve(convin.c * kernel.h * kernel.w);
  for(int c = 0; c < convin.c; ++c) {
    for(int dy = 0; dy < kernel.h; ++dy) {
      for(int dx = 0; dx < kernel.w; ++dx) {
        int kx = x + dx;
        int ky = y + dy;
        if (kx < 0 || ky < 0 || kx >= convin.w || ky >= convin.h) continue;
        xs.push_back(convin(c, ky, kx) * kernel(c, dy, dx));
      }
    }
  }

  return autodiff::reverse::sum(xs);
}

/// Each ndarray_t<T> in W represents an output channel.
/// const ndarray_t<T> &b has the same dimensions with the output.
template<typename T>
ndarray_t<T> conv2d(ndarray_t<T>& convin, std::vector<ndarray_t<T>>& W, ndarray_t<T>& b) {
  ndarray_t<T> convout(W.size(), convin.h, convin.w);
  for(int c = 0; c < W.size(); ++c) {
    for(int y = 0; y < convout.h; ++y) {
      for(int x = 0; x < convout.w; ++x) {
        convout(c, y, x) = conv2d_at(convin, W[c], y, x);
      }
    }
  }
  assert(convout.c == b.c);
  assert(convout.h == b.h);
  assert(convout.w == b.w);
  convout.v = convout.v + b.v;
  return convout;
}

template<typename T>
ndarray_t<T> conv2d_layer(ndarray_t<T>&x, std::vector<ndarray_t<T>> W, ndarray_t<T>& b, VectorXtvar<T>(f)(const VectorXtvar<T>&)) {
  ndarray_t<T> convout = conv2d(x, W, b);
  convout.v = f(convout.v);
  return convout;
}

template<typename T>
ndarray_t<T> maxpooling_2d(ndarray_t<T>& a, int sx, int sy) {
  ndarray_t<T> ret(a.c, a.h / sy, a.w / sx);
  for(int c = 0; c < ret.c; ++c) {
    for(int y = 0; y < ret.h; ++y) {
      for (int x = 0; x < ret.w; ++x) {
        T max_val = a(c, y * sy, x * sx).expr->val;
        int my = y * sy;
        int mx = x * sx;
        for(int dy = 0; dy < sy; ++dy) {
          for (int dx = 0; dx < sx; ++dx) {
            T cur = a(c, y * sy + dy, x * sx + dx).expr->val;
            if (cur > max_val) {
              max_val = cur;
              my = y * sy + dy;
              mx = x * sx + dx;
            }
          }
        }
        ret(c, y, x) = a(c, my, mx);
      }
    }
  }
  return ret;
}

template<typename T>
void dropout(VectorXtvar<T>& x, const double p) {
  const int len = x.size();
  const int ndrop = (int)(len * p);
  std::default_random_engine rng;
  std::uniform_int_distribution<int> dist(0, len-1);

  std::unordered_set<int> drop_idx;
  drop_idx.reserve(ndrop);
  while(drop_idx.size() < ndrop) {
    drop_idx.insert(dist(rng));
  }

  for(const auto &i: drop_idx) {
    x[i] = autodiff::reverse::constant<T>(T(0.0));
  }
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

