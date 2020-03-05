#pragma once

#include <cinttypes>
#include <limits>
#include <iostream>

namespace qnum {

template <typename T> struct number_traits { };
template <> struct number_traits<int8_t> { 
  typedef int16_t T2x; 
  typedef uint8_t Tu;
};
template <> struct number_traits<int16_t> { 
  typedef int32_t T2x; 
  typedef uint16_t Tu;
};
template <> struct number_traits<int32_t> { 
  typedef int64_t T2x; 
  typedef uint32_t Tu;
};

/// A Q-Space number consists of 3 parts:
/// - s: the sign bit
/// - e: the extension component, unsigned integer
/// - d: the significant compoment, unsigned integer
/// Note, both the extension and the significant are unsigned. There's only one sign bit.
/// The space is asymmetric with respect to zero. There's only a positive zero, and 
/// instead of negative zero, an extra epsilon in the negative dynamic range.
template <typename T>
struct qspace_number_t
{
  static constexpr bool saturate_ex = false;
  using T2x = typename number_traits<T>::T2x;
  using Tu = typename number_traits<T>::Tu;
  T val;

  qspace_number_t(): val(0){}
  qspace_number_t(double v) {
    val = static_cast<T>(v / (1 + ext_max()) * T_max());
    // TODO saturate ex
  }

  qspace_number_t next() const {
    qspace_number_t ret;
    ret.val = val + 1;
    return ret;
  }

  qspace_number_t prev() const {
    qspace_number_t ret;
    ret.val = val - 1;
    return ret;
  }

  double to_double() const {
    return static_cast<double>(val) / T_max() * (1+ext_max());
  }

  static T saturate(const T2x& v) {
    if (v > T_max()) return T_max();
    if (v < T_min()) return T_min();
    return static_cast<T>(v);
  }

  static constexpr T T_max() { return std::numeric_limits<T>::max(); }
  static constexpr T T_min() { return std::numeric_limits<T>::min(); }
  static constexpr int ext_bits() ;
  static constexpr int joint_bits() { return std::numeric_limits<T>::digits; }
  static constexpr int frac_bits() { return joint_bits() - ext_bits(); }
  static constexpr int ext_max() { return (1 << ext_bits()) - 1; }
  static constexpr int frac_max() { return (1 << frac_bits()) - 1; }
  static constexpr int K() { return 1 << (frac_bits() - 1); }
};

template <typename T>
std::ostream& operator << (std::ostream& os, const qspace_number_t<T>& qnum) {
  return os << qnum.to_double();
}

template <typename T>
qspace_number_t<T> operator + (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  qspace_number_t<T> ret;
  ret.val = lhs.val + rhs.val;
  return ret;
}

template <typename T>
qspace_number_t<T> operator - (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  qspace_number_t<T> ret;
  ret.val = lhs.val - rhs.val;
  return ret;
}

template <typename T>
qspace_number_t<T> operator * (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  qspace_number_t<T> ret;
  typename number_traits<T>::T2x tmp = static_cast<typename number_traits<T>::T2x>(lhs.val) * static_cast<typename number_traits<T>::T2x>(rhs.val);
  tmp += qspace_number_t<T>::K();
  ret.val = qspace_number_t<T>::saturate(tmp >> qspace_number_t<T>::frac_bits());
  return ret;
}

template <typename T>
qspace_number_t<T> operator / (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  qspace_number_t<T> ret;
  // pre-scaling up
  T2x tmp = static_cast<typename number_traits<T>::T2x>(lhs.val) << qspace_number_t<T>::frac_bits();
  // rounding
  if ((tmp >= 0 && rhs.val >= 0) || (tmp < 0 && rhs.val < 0)) {
    tmp += rhs.val / 2;
  } else {
    tmp -= rhs.val / 2;
  }
  ret.val = static_cast<T>(tmp / rhs.val);
  return ret;
}

// XXX arbitrary values..
template <> constexpr int qspace_number_t<int8_t>::ext_bits() { return 0; }
template <> constexpr int qspace_number_t<int16_t>::ext_bits() { return 1; }
template <> constexpr int qspace_number_t<int32_t>::ext_bits() { return 2; }

using qnum8_t  = qspace_number_t<int8_t>;
using qnum16_t = qspace_number_t<int16_t>;
using qnum32_t = qspace_number_t<int32_t>;

}
