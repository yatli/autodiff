//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// C++ includes
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

/// autodiff namespace where @ref var and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
namespace reverse {

template<typename T> struct Expr;
template<typename T> struct ParameterExpr;
template<typename T> struct VariableExpr;
template<typename T> struct ConstantExpr;
template<typename T> struct UnaryExpr;
template<typename T> struct NegativeExpr;
template<typename T> struct BinaryExpr;
template<typename T> struct AddExpr;
template<typename T> struct SubExpr;
template<typename T> struct MulExpr;
template<typename T> struct DivExpr;
template<typename T> struct SinExpr;
template<typename T> struct CosExpr;
template<typename T> struct TanExpr;
template<typename T> struct SinhExpr;
template<typename T> struct CoshExpr;
template<typename T> struct TanhExpr;
template<typename T> struct ArcSinExpr;
template<typename T> struct ArcCosExpr;
template<typename T> struct ArcTanExpr;
template<typename T> struct ExpExpr;
template<typename T> struct LogExpr;
template<typename T> struct Log10Expr;
template<typename T> struct PowExpr;
template<typename T> struct SqrtExpr;
template<typename T> struct AbsExpr;
template<typename T> struct ErfExpr;

template<typename T>
using ExprPtr = std::shared_ptr<const Expr<T>>;

template<typename T>
using DerivativesMap = std::unordered_map<const Expr<T>*, T>;
template<typename T>
using DerivativesMapX = std::unordered_map<const Expr<T>*, ExprPtr<T>>;

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> constant(T val);

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> operator+(const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator-(const ExprPtr<T>& r);

template<typename T>
ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r);

template<typename T>
ExprPtr<T> operator+(T l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator-(T l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator*(T l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> operator/(T l, const ExprPtr<T>& r);

template<typename T>
ExprPtr<T> operator+(const ExprPtr<T>& l, T r);
template<typename T>
ExprPtr<T> operator-(const ExprPtr<T>& l, T r);
template<typename T>
ExprPtr<T> operator*(const ExprPtr<T>& l, T r);
template<typename T>
ExprPtr<T> operator/(const ExprPtr<T>& l, T r);

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> sin(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> cos(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> tan(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> asin(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> acos(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> atan(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> sinh(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> cosh(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> tanh(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> exp(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> log(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> log10(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> pow(T l, const ExprPtr<T>& r);
template<typename T>
ExprPtr<T> pow(const ExprPtr<T>& l, T r);
template<typename T>
ExprPtr<T> sqrt(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> abs(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> abs2(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> conj(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> real(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> imag(const ExprPtr<T>& x);
template<typename T>
ExprPtr<T> erf(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
bool operator==(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
bool operator!=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
bool operator<=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
bool operator>=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
bool operator<(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T>
bool operator>(const ExprPtr<T>& l, const ExprPtr<T>& r);

template<typename T>
bool operator==(T l, const ExprPtr<T>& r);
template<typename T>
bool operator!=(T l, const ExprPtr<T>& r);
template<typename T>
bool operator<=(T l, const ExprPtr<T>& r);
template<typename T>
bool operator>=(T l, const ExprPtr<T>& r);
template<typename T>
bool operator<(T l, const ExprPtr<T>& r);
template<typename T>
bool operator>(T l, const ExprPtr<T>& r);

template<typename T>
bool operator==(const ExprPtr<T>& l, T r);
template<typename T>
bool operator!=(const ExprPtr<T>& l, T r);
template<typename T>
bool operator<=(const ExprPtr<T>& l, T r);
template<typename T>
bool operator>=(const ExprPtr<T>& l, T r);
template<typename T>
bool operator<(const ExprPtr<T>& l, T r);
template<typename T>
bool operator>(const ExprPtr<T>& l, T r);

template <typename T>
struct Expr
{
    /// The numerical value of this expression.
    T val;

    /// Construct an Expr object with given numerical value.
    explicit Expr(T val) : val(val) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression.
    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression (as an expression).
    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const = 0;
};

template <typename T>
struct ParameterExpr : Expr<T>
{
    using Expr<T>::Expr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second += wprime;
        else derivatives.insert({ this, wprime });
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second = it->second + wprime;
        else derivatives.insert({ this, wprime });
    }
};

template <typename T>
struct VariableExpr : Expr<T>
{
    ExprPtr<T> expr;

    VariableExpr(const ExprPtr<T>& expr) : Expr(expr->val), expr(expr) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second += wprime;
        else derivatives.insert({ this, wprime });
        expr->propagate(derivatives, wprime);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second = it->second + wprime;
        else derivatives.insert({ this, wprime });
        expr->propagate(derivatives, wprime);
    }
};

template <typename T>
struct ConstantExpr : Expr<T>
{
    using Expr<T>::Expr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {}

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {}
};

template <typename T>
struct UnaryExpr : Expr<T>
{
    ExprPtr<T> x;

    UnaryExpr(T val, const ExprPtr<T>& x) : Expr(val), x(x) {}
};

template <typename T>
struct NegativeExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::UnaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, -wprime);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, -wprime);
    }
};

template <typename T>
struct BinaryExpr : Expr<T>
{
    ExprPtr<T> l, r;

    BinaryExpr(T val, const ExprPtr<T>& l, const ExprPtr<T>& r) : Expr(val), l(l), r(r) {}
};

template <typename T>
struct AddExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        l->propagate(derivatives, wprime);
        r->propagate(derivatives, wprime);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        l->propagate(derivatives, wprime);
        r->propagate(derivatives, wprime);
    }
};

template <typename T>
struct SubExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        l->propagate(derivatives,  wprime);
        r->propagate(derivatives, -wprime);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        l->propagate(derivatives,  wprime);
        r->propagate(derivatives, -wprime);
    }
};

template <typename T>
struct MulExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        l->propagate(derivatives, wprime * r->val);
        r->propagate(derivatives, wprime * l->val);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        l->propagate(derivatives, wprime * r);
        r->propagate(derivatives, wprime * l);
    }
};

template <typename T>
struct DivExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto aux1 = T(1.0) / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(derivatives, wprime * aux1);
        r->propagate(derivatives, wprime * aux2);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto aux1 = T(1.0) / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagate(derivatives, wprime * aux1);
        r->propagate(derivatives, wprime * aux2);
    }
};

template <typename T>
struct SinExpr : UnaryExpr<T>
{
    SinExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime * std::cos(x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime * cos(x));
    }
};

template <typename T>
struct CosExpr : UnaryExpr<T>
{
    CosExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, -wprime * std::sin(x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, -wprime * sin(x));
    }
};

template <typename T>
struct TanExpr : UnaryExpr<T>
{
    TanExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto aux = 1.0 / std::cos(x->val);
        x->propagate(derivatives, wprime * aux * aux);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto aux = 1.0 / cos(x);
        x->propagate(derivatives, wprime * aux * aux);
    }
};

template <typename T>
struct SinhExpr : UnaryExpr<T>
{
    SinhExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime * std::cosh(x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime * cosh(x));
    }
};

template <typename T>
struct CoshExpr : UnaryExpr<T>
{
    CoshExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime * std::sinh(x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime * sinh(x));
    }
};

template <typename T>
struct TanhExpr : UnaryExpr<T>
{
    TanhExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto aux = 1.0 / std::cosh(x->val);
        x->propagate(derivatives, wprime * aux * aux);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto aux = 1.0 / cosh(x);
        x->propagate(derivatives, wprime * aux * aux);
    }
};

template <typename T>
struct ArcSinExpr : UnaryExpr<T>
{
    ArcSinExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime / sqrt(1.0 - x * x));
    }
};

template <typename T>
struct ArcCosExpr : UnaryExpr<T>
{
    ArcCosExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, -wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, -wprime / sqrt(1.0 - x * x));
    }
};

template <typename T>
struct ArcTanExpr : UnaryExpr<T>
{
    ArcTanExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime / (1.0 + x->val * x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime / (1.0 + x * x));
    }
};

template <typename T>
struct ExpExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::UnaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime * val);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime * exp(x));
    }
};

template <typename T>
struct LogExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::UnaryExpr;

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime / x->val);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime / x);
    }
};

template <typename T>
struct Log10Expr : UnaryExpr<T>
{
    constexpr static double ln10 = 2.3025850929940456840179914546843;

    Log10Expr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime / (ln10 * x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime / (ln10 * x));
    }
};

template <typename T>
struct PowExpr : BinaryExpr<T>
{
    T log_l;

    PowExpr(T val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r), log_l(std::log(l->val)) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * val;
        l->propagate(derivatives, aux * rval / lval);
        r->propagate(derivatives, aux * std::log(lval));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto aux = wprime * pow(l, r - 1.0);
        l->propagate(derivatives, aux * r);
        r->propagate(derivatives, aux *l * log(l));
    }
};

template <typename T>
struct PowConstantLeftExpr : BinaryExpr<T>
{
    PowConstantLeftExpr(T val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        r->propagate(derivatives, wprime * val * std::log(l->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        r->propagate(derivatives, wprime * pow(l, r) * log(l));
    }
};

template <typename T>
struct PowConstantRightExpr : BinaryExpr<T>
{
    PowConstantRightExpr(T val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        l->propagate(derivatives, wprime * val * r->val / l->val);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        l->propagate(derivatives, wprime * pow(l, r - 1.0) * r);
    }
};

template <typename T>
struct SqrtExpr : UnaryExpr<T>
{
    SqrtExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime / (2.0 * std::sqrt(x->val)));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime / (2.0 * sqrt(x)));
    }
};

template <typename T>
struct AbsExpr : UnaryExpr<T>
{
    AbsExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        x->propagate(derivatives, wprime * std::copysign(1.0, x->val));
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        x->propagate(derivatives, wprime * std::copysign(1.0, x->val));
    }
};

template <typename T>
struct ErfExpr : UnaryExpr<T>
{
    constexpr static auto sqrt_pi = 1.7724538509055160272981674833411451872554456638435;

    ErfExpr(T val, const ExprPtr<T>& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap<T>& derivatives, T wprime) const
    {
        const auto aux = 2.0/sqrt_pi * std::exp(-(x->val)*(x->val));
        x->propagate(derivatives, wprime * aux);
    }

    virtual void propagate(DerivativesMapX<T>& derivatives, const ExprPtr<T>& wprime) const
    {
        const auto aux = 2.0/sqrt_pi * exp(-x*x);
        x->propagate(derivatives, wprime * aux);
    }
};

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> constant(T val) { return std::make_shared<ConstantExpr<T>>(val); }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> operator+(const ExprPtr<T>& r) { return r; }
template <typename T>
inline ExprPtr<T> operator-(const ExprPtr<T>& r) { return std::make_shared<NegativeExpr<T>>(-r->val, r); }

template <typename T>
inline ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<AddExpr<T>>(l->val + r->val, l, r); }
template <typename T>
inline ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<SubExpr<T>>(l->val - r->val, l, r); }
template <typename T>
inline ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<MulExpr<T>>(l->val * r->val, l, r); }
template <typename T>
inline ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<DivExpr<T>>(l->val / r->val, l, r); }

template <typename T>
inline ExprPtr<T> operator+(T l, const ExprPtr<T>& r) { return constant(l) + r; }
template <typename T>
inline ExprPtr<T> operator-(T l, const ExprPtr<T>& r) { return constant(l) - r; }
template <typename T>
inline ExprPtr<T> operator*(T l, const ExprPtr<T>& r) { return constant(l) * r; }
template <typename T>
inline ExprPtr<T> operator/(T l, const ExprPtr<T>& r) { return constant(l) / r; }

template <typename T>
inline ExprPtr<T> operator+(const ExprPtr<T>& l, T r) { return l + constant(r); }
template <typename T>
inline ExprPtr<T> operator-(const ExprPtr<T>& l, T r) { return l - constant(r); }
template <typename T>
inline ExprPtr<T> operator*(const ExprPtr<T>& l, T r) { return l * constant(r); }
template <typename T>
inline ExprPtr<T> operator/(const ExprPtr<T>& l, T r) { return l / constant(r); }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> sin(const ExprPtr<T>& x) { return std::make_shared<SinExpr<T>>(std::sin(x->val), x); }
template <typename T>
inline ExprPtr<T> cos(const ExprPtr<T>& x) { return std::make_shared<CosExpr<T>>(std::cos(x->val), x); }
template <typename T>
inline ExprPtr<T> tan(const ExprPtr<T>& x) { return std::make_shared<TanExpr<T>>(std::tan(x->val), x); }
template <typename T>
inline ExprPtr<T> asin(const ExprPtr<T>& x) { return std::make_shared<ArcSinExpr<T>>(std::asin(x->val), x); }
template <typename T>
inline ExprPtr<T> acos(const ExprPtr<T>& x) { return std::make_shared<ArcCosExpr<T>>(std::acos(x->val), x); }
template <typename T>
inline ExprPtr<T> atan(const ExprPtr<T>& x) { return std::make_shared<ArcTanExpr<T>>(std::atan(x->val), x); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> sinh(const ExprPtr<T>& x) { return std::make_shared<SinhExpr<T>>(std::sinh(x->val), x); }
template <typename T>
inline ExprPtr<T> cosh(const ExprPtr<T>& x) { return std::make_shared<CoshExpr<T>>(std::cosh(x->val), x); }
template <typename T>
inline ExprPtr<T> tanh(const ExprPtr<T>& x) { return std::make_shared<TanhExpr<T>>(std::tanh(x->val), x); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> exp(const ExprPtr<T>& x) { return std::make_shared<ExpExpr<T>>(std::exp(x->val), x); }
template <typename T>
inline ExprPtr<T> log(const ExprPtr<T>& x) { return std::make_shared<LogExpr<T>>(std::log(x->val), x); }
template <typename T>
inline ExprPtr<T> log10(const ExprPtr<T>& x) { return std::make_shared<Log10Expr<T>>(std::log10(x->val), x); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<PowExpr<T>>(std::pow(l->val, r->val), l, r); }
template <typename T>
inline ExprPtr<T> pow(T l, const ExprPtr<T>& r) { return std::make_shared<PowConstantLeftExpr<T>>(std::pow(l, r->val), constant(l), r); }
template <typename T>
inline ExprPtr<T> pow(const ExprPtr<T>& l, T r) { return std::make_shared<PowConstantRightExpr<T>>(std::pow(l->val, r), l, constant(r)); }
template <typename T>
inline ExprPtr<T> sqrt(const ExprPtr<T>& x) { return std::make_shared<SqrtExpr<T>>(std::sqrt(x->val), x); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> abs(const ExprPtr<T>& x) { return std::make_shared<AbsExpr<T>>(std::abs(x->val), x); }
template <typename T>
inline ExprPtr<T> abs2(const ExprPtr<T>& x) { return x * x; }
template <typename T>
inline ExprPtr<T> conj(const ExprPtr<T>& x) { return x; }
template <typename T>
inline ExprPtr<T> real(const ExprPtr<T>& x) { return x; }
template <typename T>
inline ExprPtr<T> imag(const ExprPtr<T>& x) { return constant(0.0); }
template <typename T>
inline ExprPtr<T> erf(const ExprPtr<T>& x) { return std::make_shared<ErfExpr<T>>(std::erf(x->val), x); }

//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------
template <typename T>
inline bool operator==(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val == r->val; }
template <typename T>
inline bool operator!=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val != r->val; }
template <typename T>
inline bool operator<=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val <= r->val; }
template <typename T>
inline bool operator>=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val >= r->val; }
template <typename T>
inline bool operator<(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val < r->val; }
template <typename T>
inline bool operator>(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val > r->val; }

template <typename T>
inline bool operator==(T l, const ExprPtr<T>& r) { return l == r->val; }
template <typename T>
inline bool operator!=(T l, const ExprPtr<T>& r) { return l != r->val; }
template <typename T>
inline bool operator<=(T l, const ExprPtr<T>& r) { return l <= r->val; }
template <typename T>
inline bool operator>=(T l, const ExprPtr<T>& r) { return l >= r->val; }
template <typename T>
inline bool operator<(T l, const ExprPtr<T>& r) { return l < r->val; }
template <typename T>
inline bool operator>(T l, const ExprPtr<T>& r) { return l > r->val; }

template <typename T>
inline bool operator==(const ExprPtr<T>& l, T r) { return l->val == r; }
template <typename T>
inline bool operator!=(const ExprPtr<T>& l, T r) { return l->val != r; }
template <typename T>
inline bool operator<=(const ExprPtr<T>& l, T r) { return l->val <= r; }
template <typename T>
inline bool operator>=(const ExprPtr<T>& l, T r) { return l->val >= r; }
template <typename T>
inline bool operator<(const ExprPtr<T>& l, T r) { return l->val < r; }
template <typename T>
inline bool operator>(const ExprPtr<T>& l, T r) { return l->val > r; }

} // namespace reverse

using namespace reverse;

/// The autodiff variable type used for automatic differentiation.
template <typename T>
struct var
{
    /// The pointer to the expression tree of variable operations
    ExprPtr<T> expr;

    /// Construct a default var object variable
    var() : var(0.0) {}

    /// Construct a var object variable with given value
    var(T val) : expr(std::make_shared<ParameterExpr<T>>(val)) {}

    /// Construct a var object variable with given int
    var(typename std::enable_if_t<!std::is_same_v<T,int>, int> val) : expr(std::make_shared<ParameterExpr<T>>(static_cast<T>(val))) { }

    /// Construct a var object variable with given expression
    var(const ExprPtr<T>& expr) : expr(std::make_shared<VariableExpr<T>>(expr)) {}

    /// Implicitly convert this var object variable into an expression pointer
    operator ExprPtr<T>() const { return expr; }

    /// Explicitly convert this var object variable into a T value
    explicit operator T() const { return expr->val; }

	// Arithmetic-assignment operators
    var& operator+=(const ExprPtr<T>& other) { expr = expr + other; return *this; }
    var& operator-=(const ExprPtr<T>& other) { expr = expr - other; return *this; }
    var& operator*=(const ExprPtr<T>& other) { expr = expr * other; return *this; }
    var& operator/=(const ExprPtr<T>& other) { expr = expr / other; return *this; }
    var& operator+=(T other) { expr = expr + constant(other); return *this; }
    var& operator-=(T other) { expr = expr - constant(other); return *this; }
    var& operator*=(T other) { expr = expr * constant(other); return *this; }
    var& operator/=(T other) { expr = expr / constant(other); return *this; }
};

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
template <typename T>
inline bool operator==(const var<T>& l, const var<T>& r) { return l.expr == r.expr; }
template <typename T>
inline bool operator!=(const var<T>& l, const var<T>& r) { return l.expr != r.expr; }
template <typename T>
inline bool operator<=(const var<T>& l, const var<T>& r) { return l.expr <= r.expr; }
template <typename T>
inline bool operator>=(const var<T>& l, const var<T>& r) { return l.expr >= r.expr; }
template <typename T>
inline bool operator<(const var<T>& l, const var<T>& r) { return l.expr < r.expr; }
template <typename T>
inline bool operator>(const var<T>& l, const var<T>& r) { return l.expr > r.expr; }

template <typename T>
inline bool operator==(T l, const var<T>& r) { return l == r.expr; }
template <typename T>
inline bool operator!=(T l, const var<T>& r) { return l != r.expr; }
template <typename T>
inline bool operator<=(T l, const var<T>& r) { return l <= r.expr; }
template <typename T>
inline bool operator>=(T l, const var<T>& r) { return l >= r.expr; }
template <typename T>
inline bool operator<(T l, const var<T>& r) { return l < r.expr; }
template <typename T>
inline bool operator>(T l, const var<T>& r) { return l > r.expr; }

template <typename T>
inline bool operator==(const var<T>& l, T r) { return l.expr == r; }
template <typename T>
inline bool operator!=(const var<T>& l, T r) { return l.expr != r; }
template <typename T>
inline bool operator<=(const var<T>& l, T r) { return l.expr <= r; }
template <typename T>
inline bool operator>=(const var<T>& l, T r) { return l.expr >= r; }
template <typename T>
inline bool operator<(const var<T>& l, T r) { return l.expr < r; }
template <typename T>
inline bool operator>(const var<T>& l, T r) { return l.expr > r; }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline const ExprPtr<T>& operator+(const var<T>& r) { return r.expr; }
template <typename T>
inline ExprPtr<T> operator-(const var<T>& r) { return -r.expr; }

template <typename T>
inline ExprPtr<T> operator+(const var<T>& l, const var<T>& r) { return l.expr + r.expr; }
template <typename T>
inline ExprPtr<T> operator-(const var<T>& l, const var<T>& r) { return l.expr - r.expr; }
template <typename T>
inline ExprPtr<T> operator*(const var<T>& l, const var<T>& r) { return l.expr * r.expr; }
template <typename T>
inline ExprPtr<T> operator/(const var<T>& l, const var<T>& r) { return l.expr / r.expr; }

template <typename T>
inline ExprPtr<T> operator+(const ExprPtr<T>& l, const var<T>& r) { return l + r.expr; }
template <typename T>
inline ExprPtr<T> operator-(const ExprPtr<T>& l, const var<T>& r) { return l - r.expr; }
template <typename T>
inline ExprPtr<T> operator*(const ExprPtr<T>& l, const var<T>& r) { return l * r.expr; }
template <typename T>
inline ExprPtr<T> operator/(const ExprPtr<T>& l, const var<T>& r) { return l / r.expr; }

template <typename T>
inline ExprPtr<T> operator+(const var<T>& l, const ExprPtr<T>& r) { return l.expr + r; }
template <typename T>
inline ExprPtr<T> operator-(const var<T>& l, const ExprPtr<T>& r) { return l.expr - r; }
template <typename T>
inline ExprPtr<T> operator*(const var<T>& l, const ExprPtr<T>& r) { return l.expr * r; }
template <typename T>
inline ExprPtr<T> operator/(const var<T>& l, const ExprPtr<T>& r) { return l.expr / r; }

template <typename T>
inline ExprPtr<T> operator+(T l, const var<T>& r) { return l + r.expr; }
template <typename T>
inline ExprPtr<T> operator-(T l, const var<T>& r) { return l - r.expr; }
template <typename T>
inline ExprPtr<T> operator*(T l, const var<T>& r) { return l * r.expr; }
template <typename T>
inline ExprPtr<T> operator/(T l, const var<T>& r) { return l / r.expr; }

template <typename T>
inline ExprPtr<T> operator+(const var<T>& l, T r) { return l.expr + r; }
template <typename T>
inline ExprPtr<T> operator-(const var<T>& l, T r) { return l.expr - r; }
template <typename T>
inline ExprPtr<T> operator*(const var<T>& l, T r) { return l.expr * r; }
template <typename T>
inline ExprPtr<T> operator/(const var<T>& l, T r) { return l.expr / r; }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> sin(const var<T>& x) { return sin(x.expr); }
template <typename T>
inline ExprPtr<T> cos(const var<T>& x) { return cos(x.expr); }
template <typename T>
inline ExprPtr<T> tan(const var<T>& x) { return tan(x.expr); }
template <typename T>
inline ExprPtr<T> asin(const var<T>& x) { return asin(x.expr); }
template <typename T>
inline ExprPtr<T> acos(const var<T>& x) { return acos(x.expr); }
template <typename T>
inline ExprPtr<T> atan(const var<T>& x) { return atan(x.expr); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> sinh(const var<T>& x) { return sinh(x.expr); }
template <typename T>
inline ExprPtr<T> cosh(const var<T>& x) { return cosh(x.expr); }
template <typename T>
inline ExprPtr<T> tanh(const var<T>& x) { return tanh(x.expr); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> exp(const var<T>& x) { return exp(x.expr); }
template <typename T>
inline ExprPtr<T> log(const var<T>& x) { return log(x.expr); }
template <typename T>
inline ExprPtr<T> log10(const var<T>& x) { return log10(x.expr); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> pow(const var<T>& l, const var<T>& r) { return pow(l.expr, r.expr); }
template <typename T>
inline ExprPtr<T> pow(T l, const var<T>& r) { return pow(l, r.expr); }
template <typename T>
inline ExprPtr<T> pow(const var<T>& l, T r) { return pow(l.expr, r); }
template <typename T>
inline ExprPtr<T> sqrt(const var<T>& x) { return sqrt(x.expr); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var<T>)
//------------------------------------------------------------------------------
template <typename T>
inline ExprPtr<T> abs(const var<T>& x) { return abs(x.expr); }
template <typename T>
inline ExprPtr<T> abs2(const var<T>& x) { return abs2(x.expr); }
template <typename T>
inline ExprPtr<T> conj(const var<T>& x) { return conj(x.expr); }
template <typename T>
inline ExprPtr<T> real(const var<T>& x) { return real(x.expr); }
template <typename T>
inline ExprPtr<T> imag(const var<T>& x) { return imag(x.expr); }
template <typename T>
inline ExprPtr<T> erf(const var<T>& x) { return erf(x.expr); }

/// Return the value of a variable x.
template <typename T>
inline T val(const var<T>& x)
{
    return x.expr->val;
}

template <typename T>
using Derivatives = std::function<T(const var<T>&)>;
template <typename T>
using DerivativesX = std::function<var<T>(const var<T>&)>;

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
inline Derivatives<T> derivatives(const var<T>& y)
{
    DerivativesMap<T> map;

    y.expr->propagate(map, 1.0);

    auto fn = [=](const var<T>& x)
    {
        const auto it = map.find(x.expr.get());
        return it != map.end() ? it->second : 0.0;
    };

    return fn;
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
inline DerivativesX<T> derivativesx(const var<T>& y)
{
    DerivativesMapX<T> map;

    y.expr->propagate(map, constant(1.0));

    auto fn = [=](const var<T>& x)
    {
        const auto it = map.find(x.expr.get());
        return it != map.end() ? it->second : constant(0.0);
    };

    return fn;
}

/// Output a var object variable to the output stream.
template<typename T>
inline std::ostream& operator<<(std::ostream& out, const var<T>& x)
{
    out << autodiff::val<T>(x);
    return out;
}

} // namespace autodiff
