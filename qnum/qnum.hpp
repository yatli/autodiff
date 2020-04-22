#pragma once

#include <cinttypes>
#include <limits>
#include <iostream>
#include <cmath>

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
template <> struct number_traits<int64_t> { 
  typedef int64_t T2x; 
  typedef uint64_t Tu;
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
  using T2x = typename number_traits<T>::T2x;
  using Tu = typename number_traits<T>::Tu;
  T val;

public:

  qspace_number_t(): val(0){}
  qspace_number_t(double v) {
    auto constexpr upper = 1 + ext_max();
    if (v > upper) v = upper;
    if (v < -upper) v = -upper;
    val = static_cast<T>(v / (upper) * T_max());
  }
  qspace_number_t(const int& v) : qspace_number_t<T>(static_cast<double>(v)) { }

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

  qspace_number_t<T> neg() const {
    return from_literal(-val);
  }

  qspace_number_t<T> add(const qspace_number_t<T>& rhs) const {
    qspace_number_t<T> ret;
    T2x tmp = T2x(val) + T2x(rhs.val);
    ret.val = saturate(tmp);
    return ret;
  }

  qspace_number_t<T> sub(const qspace_number_t<T>& rhs) const {
    qspace_number_t<T> ret;
    T2x tmp = T2x(val) - T2x(rhs.val);
    ret.val = saturate(tmp);
    return ret;
  }

  qspace_number_t<T> mul(const qspace_number_t<T>& rhs) const {
    qspace_number_t<T> ret;
    T2x tmp = static_cast<T2x>(val) * static_cast<T2x>(rhs.val);
    tmp += K();
    ret.val = saturate(tmp >> frac_bits());
    return ret;
  }

  qspace_number_t<T> div(const qspace_number_t<T>& rhs) const {
    qspace_number_t<T> ret;
    // pre-scaling up
    T2x tmp = static_cast<T2x>(val) << frac_bits();
    // rounding
    if ((tmp >= 0 && rhs.val >= 0) || (tmp < 0 && rhs.val < 0)) {
      tmp += rhs.val / 2;
    } else {
      tmp -= rhs.val / 2;
    }
    ret.val = static_cast<T>(tmp / rhs.val);
    return ret;
  }

  bool operator == (const qspace_number_t<T>& rhs) const {
    return val == rhs.val;
  }

  bool operator < (const qspace_number_t<T>& rhs) const {
    return val < rhs.val;
  }

  bool operator != (const qspace_number_t<T>& rhs) const {
    return !(*this == rhs);
  }

  bool operator <= (const qspace_number_t<T>& rhs) const {
    return *this < rhs || *this == rhs;
  }

  bool operator > (const qspace_number_t<T>& rhs) const {
    return rhs < *this;
  }

  bool operator >= (const qspace_number_t<T>& rhs) const {
    return rhs <= *this;
  }

  qspace_number_t<T>& operator += (const qspace_number_t<T>& rhs) {
    *this = *this + rhs;
    return *this;
  }

  qspace_number_t<T>& operator -= (const qspace_number_t<T>& rhs) {
    *this = *this - rhs;
    return *this;
  }

  qspace_number_t<T>& operator *= (const qspace_number_t<T>& rhs) {
    *this = *this * rhs;
    return *this;
  }

  qspace_number_t<T>& operator /= (const qspace_number_t<T>& rhs) {
    *this = *this / rhs;
    return *this;
  }

  static T saturate(const T2x& v) {
    if (v > T_max()) return T_max();
    if (v < T_min()) return T_min();
    return static_cast<T>(v);
  }

  static qspace_number_t<T> from_literal(const T& t) {
    qspace_number_t<T> ret;
    ret.val = t;
    return ret;
  }

  static constexpr T T_max() { return std::numeric_limits<T>::max(); }
  static constexpr T T_min() { return std::numeric_limits<T>::min(); }
  static constexpr int ext_bits() ;
  static constexpr int joint_bits() { return std::numeric_limits<T>::digits; }
  static constexpr int frac_bits() { return joint_bits() - ext_bits(); }
  static constexpr T ext_max() { return (1 << ext_bits()) - 1; }
  static constexpr T frac_max() { return (1 << frac_bits()) - 1; }
  static constexpr int K() { return 1 << (frac_bits() - 1); } // for rounding
};

template <typename T>
std::ostream& operator << (std::ostream& os, const qspace_number_t<T>& qnum) {
  return os << qnum.to_double();
}

template <typename T>
qspace_number_t<T> operator - (const qspace_number_t<T> &x) {
  return x.neg();
}


template <typename T>
qspace_number_t<T> operator + (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  return lhs.add(rhs);
}

template <typename T>
qspace_number_t<T> operator - (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  return lhs.sub(rhs);
}

template <typename T>
qspace_number_t<T> operator * (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  return lhs.mul(rhs);
}

template <typename T>
qspace_number_t<T> operator / (const qspace_number_t<T> &lhs, const qspace_number_t<T> &rhs) {
  return lhs.div(rhs);
}

// XXX arbitrary values..
template <> constexpr int qspace_number_t<int8_t>::ext_bits() { return 1; }
template <> constexpr int qspace_number_t<int16_t>::ext_bits() { return 2; }
template <> constexpr int qspace_number_t<int32_t>::ext_bits() { return 4; }
template <> constexpr int qspace_number_t<int64_t>::ext_bits() { return 8; }

using qnum8_t  = qspace_number_t<int8_t>;
using qnum16_t = qspace_number_t<int16_t>;
using qnum32_t = qspace_number_t<int32_t>;
using qnum64_t = qspace_number_t<int64_t>;

}

namespace std
{
  using namespace qnum;

  template <> struct is_floating_point<qnum8_t> : true_type { };
  template <> struct is_floating_point<qnum16_t> : true_type { };
  template <> struct is_floating_point<qnum32_t> : true_type { };
  template <> struct is_floating_point<qnum64_t> : true_type { };

  template<typename T>
  qspace_number_t<T> log10(const qspace_number_t<T>& q) noexcept
  {
    return log10(q.to_double());
  }

  template<typename T>
  qspace_number_t<T> log(const qspace_number_t<T>& q) noexcept
  {
    return log(q.to_double());
  }

  template<typename T>
  qspace_number_t<T> exp(const qspace_number_t<T>& q) noexcept
  {
    return exp(q.to_double());
  }

  template<typename T>
  qspace_number_t<T> abs(const qspace_number_t<T>& q) noexcept
  {
    auto val = q.val;
    if (val < 0) val = -val;
    return qspace_number_t<T>::from_literal(val);
  }

  template<typename T>
  qspace_number_t<T> copysign(const qspace_number_t<T>& a, const qspace_number_t<T>& b) noexcept
  {
    auto val = a.val;
    if (b.val < 0) val = -val;
    return qspace_number_t<T>::from_literal(val);
  }

  template<typename T>
  qspace_number_t<T> copysign(const double& a, const qspace_number_t<T>& b) noexcept
  {
    qspace_number_t<T> a_ = a;
    if (b.val < 0) a_ = -a_;
    return a_;
  }
}
