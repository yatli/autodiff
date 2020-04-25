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
template <typename T, int E>
struct qspace_number_t
{
  using T2x = typename number_traits<T>::T2x;
  using Tu = typename number_traits<T>::Tu;
  using Ts = T;
  T val;

public:

  qspace_number_t(): val(0){}
  qspace_number_t(double v) {
    auto constexpr upper = 1 + ext_max();
    if (v > upper) v = upper;
    if (v < -upper) v = -upper;
    val = static_cast<T>(v / (upper) * T_max());
  }
  qspace_number_t(const int& v) : qspace_number_t<T, E>(static_cast<double>(v)) { }

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

  explicit operator double() const {
    return to_double();
  }

  qspace_number_t<T, E> neg() const {
    return from_literal(-val);
  }

  qspace_number_t<T, E> add(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    T2x tmp = T2x(val) + T2x(rhs.val);
    ret.val = saturate(tmp);
    return ret;
  }

  qspace_number_t<T, E> sub(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    T2x tmp = T2x(val) - T2x(rhs.val);
    ret.val = saturate(tmp);
    return ret;
  }

  qspace_number_t<T, E> mul(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    T2x tmp = static_cast<T2x>(val) * static_cast<T2x>(rhs.val);
    tmp += K();
    ret.val = saturate(tmp >> frac_bits());
    return ret;
  }

  qspace_number_t<T, E> div(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
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

  bool operator == (const qspace_number_t<T, E>& rhs) const {
    return val == rhs.val;
  }

  bool operator < (const qspace_number_t<T, E>& rhs) const {
    return val < rhs.val;
  }

  bool operator != (const qspace_number_t<T, E>& rhs) const {
    return !(*this == rhs);
  }

  bool operator <= (const qspace_number_t<T, E>& rhs) const {
    return *this < rhs || *this == rhs;
  }

  bool operator > (const qspace_number_t<T, E>& rhs) const {
    return rhs < *this;
  }

  bool operator >= (const qspace_number_t<T, E>& rhs) const {
    return rhs <= *this;
  }

  qspace_number_t<T, E>& operator += (const qspace_number_t<T, E>& rhs) {
    *this = *this + rhs;
    return *this;
  }

  qspace_number_t<T, E>& operator -= (const qspace_number_t<T, E>& rhs) {
    *this = *this - rhs;
    return *this;
  }

  qspace_number_t<T, E>& operator *= (const qspace_number_t<T, E>& rhs) {
    *this = *this * rhs;
    return *this;
  }

  qspace_number_t<T, E>& operator /= (const qspace_number_t<T, E>& rhs) {
    *this = *this / rhs;
    return *this;
  }

  bool saturated() const {
    return val == T_max() || val == T_min();
  }

  static T saturate(const T2x& v) {
    if (v > T_max()) return T_max();
    if (v < T_min()) return T_min();
    return static_cast<T>(v);
  }

  static qspace_number_t<T, E> from_literal(const T& t) {
    qspace_number_t<T, E> ret;
    ret.val = t;
    return ret;
  }

  static constexpr T T_max() { return std::numeric_limits<T>::max(); }
  static constexpr T T_min() { return std::numeric_limits<T>::min(); }
  static constexpr int ext_bits() { return E; }
  static constexpr int joint_bits() { return std::numeric_limits<T>::digits; }
  static constexpr int frac_bits() { return joint_bits() - ext_bits(); }
  static constexpr T ext_max() { return (1 << ext_bits()) - 1; }
  static constexpr T frac_max() { return (1 << frac_bits()) - 1; }
  static constexpr int K() { return 1 << (frac_bits() - 1); } // for rounding
};

template <typename T, int E>
std::ostream& operator << (std::ostream& os, const qspace_number_t<T, E>& qnum) {
  return os << qnum.to_double();
}

template <typename T, int E>
qspace_number_t<T, E> operator - (const qspace_number_t<T, E> &x) {
  return x.neg();
}


template <typename T, int E>
qspace_number_t<T, E> operator + (const qspace_number_t<T, E> &lhs, const qspace_number_t<T, E> &rhs) {
  return lhs.add(rhs);
}

template <typename T, int E>
qspace_number_t<T, E> operator - (const qspace_number_t<T, E> &lhs, const qspace_number_t<T, E> &rhs) {
  return lhs.sub(rhs);
}

template <typename T, int E>
qspace_number_t<T, E> operator * (const qspace_number_t<T, E> &lhs, const qspace_number_t<T, E> &rhs) {
  return lhs.mul(rhs);
}

template <typename T, int E>
qspace_number_t<T, E> operator / (const qspace_number_t<T, E> &lhs, const qspace_number_t<T, E> &rhs) {
  return lhs.div(rhs);
}

template<int E=1> using qnum8_t  = qspace_number_t<int8_t, E>;
template<int E=4> using qnum16_t = qspace_number_t<int16_t, E>;
template<int E=6>using qnum32_t  = qspace_number_t<int32_t, E>;

}

namespace std
{
  using namespace qnum;

  template <> struct is_floating_point<qnum8_t<1>> : true_type { };
  template <> struct is_floating_point<qnum16_t<1>> : true_type { };
  template <> struct is_floating_point<qnum32_t<1>> : true_type { };

  template <> struct is_floating_point<qnum8_t<2>> : true_type { };
  template <> struct is_floating_point<qnum16_t<2>> : true_type { };
  template <> struct is_floating_point<qnum32_t<2>> : true_type { };

  template <> struct is_floating_point<qnum8_t<3>> : true_type { };
  template <> struct is_floating_point<qnum16_t<3>> : true_type { };
  template <> struct is_floating_point<qnum32_t<3>> : true_type { };

  template <> struct is_floating_point<qnum8_t<4>> : true_type { };
  template <> struct is_floating_point<qnum16_t<4>> : true_type { };
  template <> struct is_floating_point<qnum32_t<4>> : true_type { };

  template <> struct is_floating_point<qnum8_t<5>> : true_type { };
  template <> struct is_floating_point<qnum16_t<5>> : true_type { };
  template <> struct is_floating_point<qnum32_t<5>> : true_type { };

  template <> struct is_floating_point<qnum8_t<6>> : true_type { };
  template <> struct is_floating_point<qnum16_t<6>> : true_type { };
  template <> struct is_floating_point<qnum32_t<6>> : true_type { };

  template <> struct is_floating_point<qnum8_t<7>> : true_type { };
  template <> struct is_floating_point<qnum16_t<7>> : true_type { };
  template <> struct is_floating_point<qnum32_t<7>> : true_type { };

  template <> struct is_floating_point<qnum8_t<8>> : true_type { };
  template <> struct is_floating_point<qnum16_t<8>> : true_type { };
  template <> struct is_floating_point<qnum32_t<8>> : true_type { };

  template <> struct is_floating_point<qnum8_t<9>> : true_type { };
  template <> struct is_floating_point<qnum16_t<9>> : true_type { };
  template <> struct is_floating_point<qnum32_t<9>> : true_type { };

  template <> struct is_floating_point<qnum8_t<10>> : true_type { };
  template <> struct is_floating_point<qnum16_t<10>> : true_type { };
  template <> struct is_floating_point<qnum32_t<10>> : true_type { };

  template <> struct is_floating_point<qnum8_t<11>> : true_type { };
  template <> struct is_floating_point<qnum16_t<11>> : true_type { };
  template <> struct is_floating_point<qnum32_t<11>> : true_type { };

  template <> struct is_floating_point<qnum8_t<12>> : true_type { };
  template <> struct is_floating_point<qnum16_t<12>> : true_type { };
  template <> struct is_floating_point<qnum32_t<12>> : true_type { };

  template<typename T, int E>
  qspace_number_t<T, E> log10(const qspace_number_t<T, E>& q) noexcept
  {
    return log10(q.to_double());
  }

  template<typename T, int E>
  qspace_number_t<T, E> log(const qspace_number_t<T, E>& q) noexcept
  {
    return log(q.to_double());
  }

  template<typename T, int E>
  qspace_number_t<T, E> exp(const qspace_number_t<T, E>& q) noexcept
  {
    return exp(q.to_double());
  }

  template<typename T, int E>
  qspace_number_t<T, E> abs(const qspace_number_t<T, E>& q) noexcept
  {
    auto val = q.val;
    if (val < 0) val = -val;
    return qspace_number_t<T, E>::from_literal(val);
  }

  template<typename T, int E>
  qspace_number_t<T, E> copysign(const qspace_number_t<T, E>& a, const qspace_number_t<T, E>& b) noexcept
  {
    auto val = a.val;
    if (b.val < 0) val = -val;
    return qspace_number_t<T, E>::from_literal(val);
  }

  template<typename T, int E>
  qspace_number_t<T, E> copysign(const double& a, const qspace_number_t<T, E>& b) noexcept
  {
    qspace_number_t<T, E> a_ = a;
    if (b.val < 0) a_ = -a_;
    return a_;
  }
}
