#pragma once

#include <cinttypes>
#include <limits>
#include <iostream>
#include <tuple>
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

/// A Q-Space number consists of 4 parts:
/// - g: the growth bit (external, not encoded in the backing integer)
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
  bool growth;

public:

  qspace_number_t(): val(0), growth(false){}
  qspace_number_t(double v) {
    auto constexpr upper = 1 + ext_max();
    auto constexpr g_upper = 1 + g_ext_max();

    if (v > upper || v < -upper) {
      growth = true;
      if (v > g_upper) v = g_upper;
      if (v < -g_upper) v = -g_upper;
      val = static_cast<T>(v / (g_upper) * T_max());
    } else {
      val = static_cast<T>(v / (upper) * T_max());
      growth = false;
    }

  }
  qspace_number_t(const int& v) : qspace_number_t<T, E>(static_cast<double>(v)) { }

  // TODO handle growth change
  qspace_number_t next() const {
    qspace_number_t ret;
    ret.val = val + 1;
    ret.growth = growth;
    return ret;
  }

  // TODO handle growth change
  qspace_number_t prev() const {
    qspace_number_t ret;
    ret.val = val - 1;
    ret.growth = growth;
    return ret;
  }

  double to_double() const {
    if (growth) {
      return static_cast<double>(val) / T_max() * (1+g_ext_max());
    } else {
      return static_cast<double>(val) / T_max() * (1+ext_max());
    }
  }

  explicit operator double() const {
    return to_double();
  }

  qspace_number_t<T, E> neg() const {
    return from_literal(-val, growth);
  }

  std::tuple<T, T, bool> align(const qspace_number_t<T, E>& rhs) const {
    T l = val;
    T r = rhs.val;
    bool g = growth || rhs.growth;
    if (g && !rhs.growth) {
      l >>= g_shift();
    } 
    if (g && !growth) {
      r >>= g_shift();
    }
    return std::make_tuple(l, r, g);
  }

  qspace_number_t<T, E> add(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    auto [l, r, g] = align(rhs);
    T2x tmp = T2x(l) + T2x(r);
    g = grow(tmp, g);
    ret.val = saturate(tmp);
    ret.growth = g;
    ret.shrink();
    return ret;
  }

  qspace_number_t<T, E> sub(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    auto [l, r, g] = align(rhs);
    T2x tmp = static_cast<T2x>(l) - static_cast<T2x>(r);
    g = grow(tmp, g);
    ret.val = saturate(tmp);
    ret.growth = g;
    ret.shrink();
    return ret;
  }

  qspace_number_t<T, E> mul(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    auto [l, r, g] = align(rhs);
    T2x tmp = static_cast<T2x>(l) * static_cast<T2x>(r);
    if (g) { 
      tmp += g_K(); 
      ret.val = saturate(tmp >> g_frac_bits());
      ret.growth = true;
    }
    else { 
      tmp += K(); 
      tmp >>= frac_bits();
      g = grow(tmp, false);
      ret.val = saturate(tmp);
      ret.growth = g;
    }
    ret.shrink();
    return ret;
  }

  qspace_number_t<T, E> div(const qspace_number_t<T, E>& rhs) const {
    qspace_number_t<T, E> ret;
    auto [l, r, g] = align(rhs);
    // pre-scaling up
    T2x tmp = static_cast<T2x>(l);
    if (g) {
      tmp <<= g_frac_bits();
    } else {
      tmp <<= frac_bits();
    }
    // rounding
    if ((tmp >= 0 && r >= 0) || (tmp < 0 && r < 0)) {
      tmp += r / 2;
    } else {
      tmp -= r/ 2;
    }
    tmp /= r;
    g = grow(tmp, g);
    ret.val = saturate(tmp);
    ret.growth = g;
    ret.shrink();
    return ret;
  }

  bool operator == (const qspace_number_t<T, E>& rhs) const {
    auto [l, r, _] = align(rhs);
    return l == r;
  }

  bool operator < (const qspace_number_t<T, E>& rhs) const {
    auto [l, r, _] = align(rhs);
    return l < r;
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

  static qspace_number_t<T, E> from_literal(const T& t, bool growth) {
    qspace_number_t<T, E> ret;
    ret.val = t;
    ret.growth = growth;
    return ret;
  }

  static bool grow(T2x& v, bool g) {
    if (g) { 
      return true;
    } else if (v > T_max() || v < T_min()) {
      v >>= g_shift();
      return true;
    } else {
      return false;
    }
  }

  void shrink() {
    if (!growth) {
      return;
    }
    if (g_threshold_min() < val && val < g_threshold_max()) {
      val <<= g_shift();
    }
  }

  // common constants
  static constexpr T T_max() { return std::numeric_limits<T>::max(); }
  static constexpr T T_min() { return std::numeric_limits<T>::min(); }
  static constexpr int joint_bits() { return std::numeric_limits<T>::digits; }

  // normal mode constants
  static constexpr int ext_bits() { return E; }
  static constexpr int frac_bits() { return joint_bits() - ext_bits(); }
  static constexpr T ext_max() { return (1 << ext_bits()) - 1; }
  static constexpr T frac_max() { return (1 << frac_bits()) - 1; }
  static constexpr int K() { return 1 << (frac_bits() - 1); } // for rounding

  // growth mode constants
  static constexpr int g_shift() { return 4; }
  static constexpr int g_ext_bits() { return E + g_shift(); }
  static constexpr int g_frac_bits() { return joint_bits() - g_ext_bits(); }
  static constexpr T g_ext_max() { return (1 << g_ext_bits()) - 1; }
  static constexpr T g_frac_max() { return (1 << g_frac_bits()) - 1; }
  static constexpr int g_K() { return 1 << (g_frac_bits() - 1); }
  static constexpr T g_threshold_max() { return T_max() >> g_shift(); }
  static constexpr T g_threshold_min() { return T_min() >> g_shift(); }
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
    return qspace_number_t<T, E>::from_literal(val, q.growth);
  }

  template<typename T, int E>
  qspace_number_t<T, E> copysign(const qspace_number_t<T, E>& a, const qspace_number_t<T, E>& b) noexcept
  {
    auto val = a.val;
    if (b.val < 0) val = -val;
    return qspace_number_t<T, E>::from_literal(val, a.growth);
  }

  template<typename T, int E>
  qspace_number_t<T, E> copysign(const double& a, const qspace_number_t<T, E>& b) noexcept
  {
    qspace_number_t<T, E> a_ = a;
    if (b.val < 0) a_ = -a_;
    return a_;
  }
}
