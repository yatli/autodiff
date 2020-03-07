#pragma once

#include <Eigen/Core>
#include "qnum.hpp"

using namespace qnum;

// Eigen specializations
namespace Eigen
{
  /// Traits specialization for qspace_number_t.
  /// See Eigen/src/Core/NumTraits.h for documentation.
  template<typename T> struct NumTraits<qspace_number_t<T>>
    : GenericNumTraits<qspace_number_t<T>>
  {
    typedef qspace_number_t<T> Real;
    typedef qspace_number_t<T> NonInteger;
    typedef qspace_number_t<T> Nested;
    typedef qspace_number_t<T> Literal;

    enum {
      IsComplex = 0,
      IsInteger = 0,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1,
      IsSigned = 1,
      RequireInitialization = 1,
    };

    static inline Real epsilon() { return Real::from_literal(1); }
    static inline Real dummy_precision() { Real v; return v; }
    static inline Real highest() { return Real::from_literal(Real::T_max()); }
    static inline Real lowest() { return Real::from_literal(Real::T_min()); }
    static inline int digits10() { return std::numeric_limits<T>::digits10; }
  };

    namespace internal {
      /// Partial specialization for random implementation for qspace numbers.
      /// See MathFunctions.h L535
      template<typename T> struct random_impl<qspace_number_t<T>>
        : random_default_impl
          <
          qspace_number_t<T>,
          NumTraits<qspace_number_t<T>>::IsComplex,
          NumTraits<qspace_number_t<T>>::IsInteger
          > 
      {
        typedef qspace_number_t<T> _Q;
        static inline _Q run(const _Q& x, const _Q& y) {
          if (x > y) return x;

          int rn = std::rand() * RAND_MAX + std::rand();
          _Q::Tu xu = static_cast<_Q::Tu>(x.val);
          _Q::Tu yu = static_cast<_Q::Tu>(y.val);
          auto ru = static_cast<_Q::Tu>(rn) % yu - xu;
          return _Q::from_literal(static_cast<T>(ru));
        }
        static inline _Q run() {
          int rn = std::rand() * RAND_MAX + std::rand();
          return _Q::from_literal(static_cast<T>(rn) >> _Q::ext_bits());
        }
      };

      template<typename T> struct random_impl<var<qspace_number_t<T>>>
        : random_default_impl
          <
          var<qspace_number_t<T>>,
          NumTraits<var<qspace_number_t<T>>>::IsComplex,
          NumTraits<var<qspace_number_t<T>>>::IsInteger
          > 
      {
        typedef qspace_number_t<T> _Q;
        static inline var<_Q> run(const var<_Q>& x, const var<_Q>& y) {
          return var(random_impl<_Q>::run(x.expr->val, y.expr->val));
        }
        static inline var<_Q> run() {
          return var(random_impl<_Q>::run());
        }
      };
    }

}

namespace std
{

  template<typename T>
  qspace_number_t<T> log10(const qspace_number_t<T>& q)
  {
    return log10(q.to_double());
  }

}