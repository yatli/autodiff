#pragma once

#include "qnum.hpp"
#include "flex.hpp"
#include <Eigen/Core>
#include "autodiff/reverse.hpp"
/// Eigen3 supporting types and helpers

// Eigen specializations
namespace Eigen
{
  using namespace qnum;
  using namespace flex;
  using namespace autodiff;
  /// Traits specialization for qspace_number_t.
  /// See Eigen/src/Core/NumTraits.h for documentation.
  template<typename T, int E> struct NumTraits<qspace_number_t<T, E>>
    : GenericNumTraits<qspace_number_t<T, E>>
  {
    typedef qspace_number_t<T, E> Real;
    typedef qspace_number_t<T, E> NonInteger;
    typedef qspace_number_t<T, E> Nested;
    typedef qspace_number_t<T, E> Literal;

    enum {
      IsComplex = 0,
      IsInteger = 0,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1,
      IsSigned = 1,
      RequireInitialization = 1,
    };

    static inline Real epsilon() { return Real::from_literal(1, false); }
    static inline Real dummy_precision() { Real v; return v; }
    static inline Real highest() { return Real::from_literal(Real::T_max(), true); }
    static inline Real lowest() { return Real::from_literal(Real::T_min(), true); }
    static inline int digits10() { return std::numeric_limits<T>::digits10; }
  };

  namespace internal {
    /// Partial specialization for random implementation for qspace numbers.
    /// See MathFunctions.h L535
    template<typename T, int E> struct random_impl<qspace_number_t<T, E>>
      : random_default_impl
        <
        qspace_number_t<T, E>,
        NumTraits<qspace_number_t<T, E>>::IsComplex,
        NumTraits<qspace_number_t<T, E>>::IsInteger
        > 
    {
      typedef qspace_number_t<T, E> _Q;
      static inline _Q run(const _Q& x, const _Q& y) {
        if (x > y) return x;

        int rn = std::rand() * RAND_MAX + std::rand();
        typename _Q::Tu xu = static_cast<typename _Q::Tu>(x.val);
        typename _Q::Tu yu = static_cast<typename _Q::Tu>(y.val);
        auto ru = static_cast<typename _Q::Tu>(rn) % yu - xu;
        return _Q::from_literal(static_cast<T>(ru), false);
      }
      static inline _Q run() {
        int rn = std::rand() * RAND_MAX + std::rand();
        return _Q::from_literal(static_cast<T>(rn) >> _Q::ext_bits(), false);
      }
    };

    template<typename T, int E> struct random_impl<Variable<qspace_number_t<T, E>>>
      : random_default_impl
        <
        Variable<qspace_number_t<T, E>>,
        NumTraits<Variable<qspace_number_t<T, E>>>::IsComplex,
        NumTraits<Variable<qspace_number_t<T, E>>>::IsInteger
        > 
    {
      typedef qspace_number_t<T, E> _Q;
      static inline Variable<_Q> run(const Variable<_Q>& x, const Variable<_Q>& y) {
        return Variable<_Q>(random_impl<_Q>::run(x.expr->val, y.expr->val));
      }
      static inline Variable<_Q> run() {
        return Variable<_Q>(random_impl<_Q>::run());
      }
    };
  }

  /// Traits specialization for flexfloat.
  /// See Eigen/src/Core/NumTraits.h for documentation.
  template<uint8_t E, uint8_t F> struct NumTraits<flexfloat<E, F>>
    : NumTraits<double>
  {
    typedef flexfloat<E, F> Real;
    typedef flexfloat<E, F> NonInteger;
    typedef flexfloat<E, F> Nested;
    typedef flexfloat<E, F> Literal;

    enum {
      RequireInitialization = 1,
    };

    static inline Real epsilon() { 
      Real v; 
      flexfloat_t ff = (flexfloat_t)v;
      flexfloat_set_bits(&ff, 1);
      return (Real)ff;
    }
    static inline Real dummy_precision() { 
      Real v; 
      return v; 
    }
    static inline Real highest() { return Real(FLT_MAX); }
    static inline Real lowest() { return Real(FLT_MIN);}
    static inline int digits10() { return 10; } // XXX wrong
  };

  namespace internal {
    /// Partial specialization for random implementation for flexfloat.
    /// See MathFunctions.h L535
    template<uint8_t E, uint8_t F> struct random_impl<flexfloat<E, F>>
      : random_default_impl
        <
        flexfloat<E, F>,
        NumTraits<flexfloat<E, F>>::IsComplex,
        NumTraits<flexfloat<E, F>>::IsInteger
        > 
    {
      typedef flexfloat<E, F> _Q;
      static inline _Q run(const _Q& x, const _Q& y) {
        if (x > y) return x;
        double rn = std::rand();
        rn = rn / RAND_MAX * (y - x) + x;
        return (_Q)rn;
      }
      static inline _Q run() {
        double rn = std::rand();
        rn = (rn / RAND_MAX) * 2.0 - 1.0;
        return (_Q)rn;
      }
    };

    template<uint8_t E, uint8_t F> struct random_impl<Variable<flexfloat<E, F>>>
      : random_default_impl
        <
        Variable<flexfloat<E, F>>,
        NumTraits<Variable<flexfloat<E, F>>>::IsComplex,
        NumTraits<Variable<flexfloat<E, F>>>::IsInteger
        > 
    {
      typedef flexfloat<E, F> _Q;
      static inline Variable<_Q> run(const Variable<_Q>& x, const Variable<_Q>& y) {
        return Variable<_Q>(random_impl<_Q>::run(x.expr->val, y.expr->val));
      }
      static inline Variable<_Q> run() {
        return Variable<_Q>(random_impl<_Q>::run());
      }
    };
  }
}

