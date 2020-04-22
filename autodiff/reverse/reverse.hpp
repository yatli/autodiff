//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

// autodiff includes
#include <autodiff/common/meta.hpp>

/// autodiff namespace where @ref Variable and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
namespace reverse {

using detail::EnableIf;
using detail::For;
using detail::isArithmetic;

template<typename T> struct Expr;
template<typename T> struct VariableExpr;
template<typename T> struct IndependentVariableExpr;
template<typename T> struct DependentVariableExpr;
template<typename T> struct ConstantExpr;
template<typename T> struct UnaryExpr;
template<typename T> struct NegativeExpr;
template<typename T> struct BinaryExpr;
template<typename T> struct SumExpr;
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
template<typename T> struct Variable;
template<typename T> struct SigmoidExpr;
template<typename T> struct ReLUExpr;

template<typename T> using ExprPtr = std::shared_ptr<Expr<T>>;

namespace traits {

template<typename T>
struct VariableValueTypeNotDefinedFor {};

template<typename T>
struct VariableValueType;

template<typename T>
struct VariableValueType { using type = std::conditional_t<isArithmetic<T>, T, VariableValueTypeNotDefinedFor<T>>; };

template<typename T>
struct VariableValueType<Variable<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableValueType<ExprPtr<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableOrder { constexpr static auto value = 0; };

template<typename T>
struct VariableOrder<Variable<T>> { constexpr static auto value = 1 + VariableOrder<T>::value; };

template<typename T>
struct isVariable { constexpr static bool value = false; };

template<typename T>
struct isVariable<Variable<T>> { constexpr static bool value = true; };

} // namespace traits

template<typename T>
using VariableValueType = typename traits::VariableValueType<T>::type;

template<typename T>
constexpr auto VariableOrder = traits::VariableOrder<T>::value;

template<typename T>
constexpr auto isVariable = traits::isVariable<T>::value;

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T>
ExprPtr<T> constant(const T& val);

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& r);
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& r);

template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r);

template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const U& l, const ExprPtr<T>& r);

template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const ExprPtr<T>& l, const U& r);

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sin(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> cos(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> tan(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> asin(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> acos(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> atan(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sinh(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> cosh(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> tanh(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> exp(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> log(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> log10(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sqrt(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const ExprPtr<T>& l, const U& r);

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> abs(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> abs2(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> conj(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> real(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> imag(const ExprPtr<T>& x);
template<typename T> ExprPtr<T> erf(const ExprPtr<T>& x);

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
template<typename T> bool operator==(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> bool operator!=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> bool operator<=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> bool operator>=(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> bool operator<(const ExprPtr<T>& l, const ExprPtr<T>& r);
template<typename T> bool operator>(const ExprPtr<T>& l, const ExprPtr<T>& r);

template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const U& l, const ExprPtr<T>& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const U& l, const ExprPtr<T>& r);

template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const ExprPtr<T>& l, const U& r);
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const ExprPtr<T>& l, const U& r);

/// The abstract type of any node type in the expression tree.
template<typename T>
struct Expr
{
    /// The value of this expression node.
    T val = {};

    /// Construct an Expr object with given value.
    explicit Expr(const T& val) : val(val) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node.
    virtual void propagate(const T& wprime) = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node (as an expression).
    virtual void propagatex(const ExprPtr<T>& wprime) = 0;

    virtual ExprPtr<T> rewrite() { return nullptr; }

    virtual const char* name() = 0;

    virtual void print(int indent) 
    { 
      std::cout << std::string(indent, ' ') << name() << std::endl;
    }
};

#define DECLARE_NAME(x) \
    virtual const char* name() { return #x; }

/// The node in the expression tree representing either an independent or dependent variable.
template<typename T>
struct VariableExpr : Expr<T>
{
    DECLARE_NAME(VariableExpr);

    /// The derivative of the root expression node with respect to this variable.
    T grad = {};

    /// The derivative of the root expression node with respect to this variable (as an expression for higher-order derivatives).
    ExprPtr<T> gradx = {};

    /// Construct a VariableExpr object with given value.
    VariableExpr(const T& val) : Expr<T>(val) {}
};

/// The node in the expression tree representing an independent variable.
template<typename T>
struct IndependentVariableExpr : VariableExpr<T>
{
    DECLARE_NAME(IndependentVariableExpr);

    // Using declarations for data members of base class
    using VariableExpr<T>::grad;
    using VariableExpr<T>::gradx;

    /// Construct an IndependentVariableExpr object with given value.
    IndependentVariableExpr(const T& val) : VariableExpr<T>(val)
    {
        gradx = constant<T>(0.0); // TODO: Check if this can be done at the seed function.
    }

    virtual void propagate(const T& wprime)
    {
        grad += wprime;
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        gradx = gradx + wprime;
    }
};

/// The node in the expression tree representing a dependent variable.
template<typename T>
struct DependentVariableExpr : VariableExpr<T>
{
    DECLARE_NAME(DependentVariableExpr);

    // Using declarations for data members of base class
    using VariableExpr<T>::grad;
    using VariableExpr<T>::gradx;

    /// The expression tree that defines how the dependent variable is calculated.
    ExprPtr<T> expr;

    /// Construct an DependentVariableExpr object with given value.
    DependentVariableExpr(const ExprPtr<T>& expr) : VariableExpr<T>(expr->val), expr(expr)
    {
        gradx = constant<T>(0.0); // TODO: Check if this can be done at the seed function.
    }

    virtual void propagate(const T& wprime)
    {
        grad += wprime;
        expr->propagate(wprime);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        gradx = gradx + wprime;
        expr->propagatex(wprime);
    }

    virtual ExprPtr<T> rewrite() {
      auto child = expr->rewrite();
      if(child) {
        expr = child;
      }
      return nullptr;
    }

    virtual void print(int indent) 
    { 
      this->Expr<T>::print(indent);
      expr->print(indent + 2);
    }
};

template<typename T>
struct ConstantExpr : Expr<T>
{
    DECLARE_NAME(ConstantExpr);

    using Expr<T>::Expr;

    virtual void propagate(const T& wprime)
    {}

    virtual void propagatex(const ExprPtr<T>& wprime)
    {}
};

template<typename T>
struct UnaryExpr : Expr<T>
{
    DECLARE_NAME(UnaryExpr);

    ExprPtr<T> x;

    UnaryExpr(const T& val, const ExprPtr<T>& x) : Expr<T>(val), x(x) {}

    virtual ExprPtr<T> rewrite() {
      auto child = x->rewrite();
      if(child) {
        x = child;
      }
      return nullptr;
    }

    virtual void print(int indent) 
    { 
      this->Expr<T>::print(indent);
      x->print(indent + 2);
    }
};

template<typename T>
struct NegativeExpr : UnaryExpr<T>
{
    DECLARE_NAME(NegativeExpr);

    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    using UnaryExpr<T>::UnaryExpr;

    virtual void propagate(const T& wprime)
    {
        x->propagate(-wprime);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(-wprime);
    }
};

template<typename T>
struct BinaryExpr : Expr<T>
{
    DECLARE_NAME(BinaryExpr);

    ExprPtr<T> l, r;

    BinaryExpr(const T& val, const ExprPtr<T>& l, const ExprPtr<T>& r) : Expr<T>(val), l(l), r(r) {}

    virtual ExprPtr<T> rewrite() {
      auto cl = l->rewrite();
      auto cr = r->rewrite();
      if(cl) {
        l = cl;
      }
      if(cr) {
        r = cr;
      }
      return nullptr;
    }

    virtual void print(int indent) 
    { 
      this->Expr<T>::print(indent);
      l->print(indent + 2);
      r->print(indent + 2);
    }
};

template<typename T>
struct SumExpr : Expr<T>
{
  DECLARE_NAME(SumExpr);

  std::vector<ExprPtr<T>> elements;

  SumExpr(const T& val, const std::vector<ExprPtr<T>> es): Expr<T>(val), elements(es) {}

  virtual void propagate(const T& wprime) 
  {
    for(auto x: elements) {
      x->propagate(wprime);
    }
  }

  virtual void propagatex(const ExprPtr<T>& wprime) 
  {
    for(auto x: elements) {
      x->propagatex(wprime);
    }
  }

  virtual ExprPtr<T> rewrite() 
  {
    for(int i = 0; i < elements.size(); ) 
    {
      auto ptr = elements[i].get();
      auto add = dynamic_cast<AddExpr<T>*>(ptr);
      auto sum = dynamic_cast<SumExpr<T>*>(ptr);
      if (!add && !sum) {
        ++i;
        continue;
      }
      if(add) {
        elements.push_back(add->l);
        elements.push_back(add->r);
      } else {
        elements.insert(elements.end(), sum->elements.begin(), sum->elements.end());
      }
      elements.erase(elements.cbegin() + i);
    }
    return nullptr;
  }

  virtual void print(int indent) 
  { 
    this->Expr<T>::print(indent);
    for(const auto &x: elements) {
      x->print(indent + 2);
    }
  }
};

template<typename T>
struct ProdExpr : Expr<T>
{
  DECLARE_NAME(ProdExpr);

  std::vector<ExprPtr<T>> elements;

  ProdExpr(const T& val, const std::vector<ExprPtr<T>> es): Expr<T>(val), elements(es) {}

  virtual void propagate(const T& wprime) 
  {
    for (auto x: elements) {
      auto w = wprime;
      for(const auto &y: elements) {
        if (y == x) continue;
        w *= y->val;
      }
      x->propagate(w);
    }
  }

  virtual void propagatex(const ExprPtr<T>& wprime) 
  {
    for (auto x: elements) {
      auto w = wprime;
      for(const auto &y: elements) {
        if (y == x) continue;
        w = w * y;
      }
      x->propagatex(w);
    }
  }

  virtual ExprPtr<T> rewrite() 
  {
    for(int i = 0; i < elements.size(); ) 
    {
      auto ptr = elements[i].get();
      auto mul = dynamic_cast<MulExpr<T>*>(ptr);
      auto prod = dynamic_cast<ProdExpr<T>*>(ptr);
      if (!mul && !prod) {
        ++i;
        continue;
      }
      if(mul) {
        elements.push_back(mul->l);
        elements.push_back(mul->r);
      } else {
        elements.insert(elements.end(), prod->elements.begin(), prod->elements.end());
      }
      elements.erase(elements.cbegin() + i);
    }
    return nullptr;
  }

  virtual void print(int indent) { 
    this->Expr<T>::print(indent);
    for(const auto &x: elements) {
      x->print(indent + 2);
    }
  }
};


template<typename T>
struct AddExpr : BinaryExpr<T>
{
    DECLARE_NAME(AddExpr);

    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(const T& wprime)
    {
        l->propagate(wprime);
        r->propagate(wprime);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        l->propagatex(wprime);
        r->propagatex(wprime);
    }

    virtual ExprPtr<T> rewrite() 
    {
      this->BinaryExpr<T>::rewrite();
      auto ladd = dynamic_cast<AddExpr<T>*>(l.get());
      auto radd = dynamic_cast<AddExpr<T>*>(r.get());
      if (ladd || radd) {
        std::vector<ExprPtr<T>> elements = {l, r};
        auto sum = std::make_shared<SumExpr<T>>(this->val, elements);
        sum->rewrite();
        return sum;
      } else {
        return nullptr;
      }
    }
};

template<typename T>
struct SubExpr : BinaryExpr<T>
{
    DECLARE_NAME(SubExpr);
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(const T& wprime)
    {
        l->propagate( wprime);
        r->propagate(-wprime);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        l->propagatex( wprime);
        r->propagatex(-wprime);
    }
};

template<typename T>
struct MulExpr : BinaryExpr<T>
{
    DECLARE_NAME(MulExpr);

    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(const T& wprime)
    {
        l->propagate(wprime * r->val);
        r->propagate(wprime * l->val);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        l->propagatex(wprime * r);
        r->propagatex(wprime * l);
    }

    virtual ExprPtr<T> rewrite() 
    {
      this->BinaryExpr<T>::rewrite();
      auto lmul = dynamic_cast<MulExpr<T>*>(l.get());
      auto rmul = dynamic_cast<MulExpr<T>*>(r.get());
      if (lmul || rmul) {
        std::vector<ExprPtr<T>> elements = {l, r};
        auto prod = std::make_shared<ProdExpr<T>>(this->val, elements);
        prod->rewrite();
        return prod;
      } else {
        return nullptr;
      }
    }
};

template<typename T>
struct DivExpr : BinaryExpr<T>
{
    DECLARE_NAME(DivExpr);
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    virtual void propagate(const T& wprime)
    {
        const auto aux1 = T(1.0) / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux1 = T(1.0) / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagatex(wprime * aux1);
        r->propagatex(wprime * aux2);
    }
};

template<typename T>
struct SinExpr : UnaryExpr<T>
{
    DECLARE_NAME(SinExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SinExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime * std::cos(x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime * cos(x));
    }
};

template<typename T>
struct CosExpr : UnaryExpr<T>
{
    DECLARE_NAME(CosExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    CosExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(-wprime * std::sin(x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(-wprime * sin(x));
    }
};

template<typename T>
struct TanExpr : UnaryExpr<T>
{
    DECLARE_NAME(TanExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    TanExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        const auto aux = 1.0 / std::cos(x->val);
        x->propagate(wprime * aux * aux);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux = 1.0 / cos(x);
        x->propagatex(wprime * aux * aux);
    }
};

template<typename T>
struct SinhExpr : UnaryExpr<T>
{
    DECLARE_NAME(SinhExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SinhExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime * std::cosh(x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime * cosh(x));
    }
};

template<typename T>
struct CoshExpr : UnaryExpr<T>
{
    DECLARE_NAME(CoshExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    CoshExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime * std::sinh(x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime * sinh(x));
    }
};

template<typename T>
struct TanhExpr : UnaryExpr<T>
{
    DECLARE_NAME(TanhExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    TanhExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        const auto aux = 1.0 / std::cosh(x->val);
        x->propagate(wprime * aux * aux);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux = 1.0 / cosh(x);
        x->propagatex(wprime * aux * aux);
    }
};

template<typename T>
struct ArcSinExpr : UnaryExpr<T>
{
    DECLARE_NAME(ArcSinExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcSinExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / sqrt(1.0 - x * x));
    }
};

template<typename T>
struct ArcCosExpr : UnaryExpr<T>
{
    DECLARE_NAME(ArcCosExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcCosExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(-wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(-wprime / sqrt(1.0 - x * x));
    }
};

template<typename T>
struct ArcTanExpr : UnaryExpr<T>
{
    DECLARE_NAME(ArcTanExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcTanExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / (1.0 + x->val * x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / (1.0 + x * x));
    }
};

template<typename T>
struct ExpExpr : UnaryExpr<T>
{
    DECLARE_NAME(ExpExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::UnaryExpr;
    using UnaryExpr<T>::val;
    using UnaryExpr<T>::x;

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime * val);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime * exp(x));
    }
};

template<typename T>
struct LogExpr : UnaryExpr<T>
{
    DECLARE_NAME(LogExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using UnaryExpr<T>::UnaryExpr;

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / x->val);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / x);
    }
};

template<typename T>
struct Log10Expr : UnaryExpr<T>
{
    DECLARE_NAME(Log10Expr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln10 = static_cast<VariableValueType<T>>(2.3025850929940456840179914546843);

    Log10Expr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / (ln10 * x->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / (ln10 * x));
    }
};

template<typename T>
struct PowExpr : BinaryExpr<T>
{
    DECLARE_NAME(PowExpr);
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    T log_l;

    PowExpr(const T& val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r), log_l(std::log(l->val)) {}

    virtual void propagate(const T& wprime)
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * val;
        l->propagate(aux * rval / lval);
        r->propagate(aux * std::log(lval));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux = wprime * pow(l, r - 1.0);
        l->propagatex(aux * r);
        r->propagatex(aux * l * log(l));
    }
};

template<typename T>
struct PowConstantLeftExpr : BinaryExpr<T>
{
    DECLARE_NAME(PowConstantLeftExpr);
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantLeftExpr(const T& val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r) {}

    virtual void propagate(const T& wprime)
    {
        r->propagate(wprime * val * std::log(l->val));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        r->propagatex(wprime * pow(l, r) * log(l));
    }
};

template<typename T>
struct PowConstantRightExpr : BinaryExpr<T>
{
    DECLARE_NAME(PowConstantRightExpr);
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantRightExpr(const T& val, const ExprPtr<T>& l, const ExprPtr<T>& r) : BinaryExpr<T>(val, l, r) {}

    virtual void propagate(const T& wprime)
    {
        l->propagate(wprime * val * r->val / l->val);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        l->propagatex(wprime * pow(l, r - 1) * r);
    }
};

template<typename T>
struct SqrtExpr : UnaryExpr<T>
{
    DECLARE_NAME(SqrtExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SqrtExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / (2.0 * std::sqrt(x->val)));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / (2.0 * sqrt(x)));
    }
};

template<typename T>
struct AbsExpr : UnaryExpr<T>
{
    DECLARE_NAME(AbsExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using U = VariableValueType<T>;

    AbsExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        if(x->val < 0.0) x->propagate(-wprime);
        else x->propagate(wprime);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        if(x->val < 0.0) x->propagatex(-wprime);
        else x->propagatex(wprime);
    }
};

template<typename T>
struct ErfExpr : UnaryExpr<T>
{
    DECLARE_NAME(ErfExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto sqrt_pi = static_cast<VariableValueType<T>>(1.7724538509055160272981674833411451872554456638435);

    ErfExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        const auto aux = 2.0/sqrt_pi * std::exp(-(x->val)*(x->val));
        x->propagate(wprime * aux);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux = 2.0/sqrt_pi * exp(-x*x);
        x->propagatex(wprime * aux);
    }
};

template <typename T>
struct SigmoidExpr : UnaryExpr<T>
{
    DECLARE_NAME(SigmoidExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    
    SigmoidExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    // XXX wrong! check this!
    virtual void propagate(const T& wprime)
    {
        x->propagate(wprime / (std::exp(x->val) + std::exp(-x->val) + T(2.0)));
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        x->propagatex(wprime / (exp(x) + exp(-x) + T(2.0)));
    }
};

template <typename T>
struct ReLUExpr : UnaryExpr<T>
{
    DECLARE_NAME(ReLUExpr);
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    
    ReLUExpr(const T& val, const ExprPtr<T>& x) : UnaryExpr<T>(val, x) {}

    virtual void propagate(const T& wprime)
    {
        const auto aux = x->val >= 0.0 ? T(1.0) : T(0.0);
        x->propagate(wprime * aux);
    }

    virtual void propagatex(const ExprPtr<T>& wprime)
    {
        const auto aux = x->val >= 0.0 ? T(1.0) : T(0.0);
        x->propagatex(wprime * aux);
    }
};

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> constant(const T& val) { return std::make_shared<ConstantExpr<T>>(val); }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& r) { return r; }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& r) { return std::make_shared<NegativeExpr<T>>(-r->val, r); }

template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<AddExpr<T>>(l->val + r->val, l, r); }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<SubExpr<T>>(l->val - r->val, l, r); }
template<typename T> ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<MulExpr<T>>(l->val * r->val, l, r); }
template<typename T> ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<DivExpr<T>>(l->val / r->val, l, r); }

template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const U& l, const ExprPtr<T>& r) { return constant<T>(l) + r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const U& l, const ExprPtr<T>& r) { return constant<T>(l) - r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const U& l, const ExprPtr<T>& r) { return constant<T>(l) * r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const U& l, const ExprPtr<T>& r) { return constant<T>(l) / r; }

template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const ExprPtr<T>& l, const U& r) { return l + constant<T>(r); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const ExprPtr<T>& l, const U& r) { return l - constant<T>(r); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const ExprPtr<T>& l, const U& r) { return l * constant<T>(r); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const ExprPtr<T>& l, const U& r) { return l / constant<T>(r); }


//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sin(const ExprPtr<T>& x) { return std::make_shared<SinExpr<T>>(std::sin(x->val), x); }
template<typename T> ExprPtr<T> cos(const ExprPtr<T>& x) { return std::make_shared<CosExpr<T>>(std::cos(x->val), x); }
template<typename T> ExprPtr<T> tan(const ExprPtr<T>& x) { return std::make_shared<TanExpr<T>>(std::tan(x->val), x); }
template<typename T> ExprPtr<T> asin(const ExprPtr<T>& x) { return std::make_shared<ArcSinExpr<T>>(std::asin(x->val), x); }
template<typename T> ExprPtr<T> acos(const ExprPtr<T>& x) { return std::make_shared<ArcCosExpr<T>>(std::acos(x->val), x); }
template<typename T> ExprPtr<T> atan(const ExprPtr<T>& x) { return std::make_shared<ArcTanExpr<T>>(std::atan(x->val), x); }


//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sinh(const ExprPtr<T>& x) { return std::make_shared<SinhExpr<T>>(std::sinh(x->val), x); }
template<typename T> ExprPtr<T> cosh(const ExprPtr<T>& x) { return std::make_shared<CoshExpr<T>>(std::cosh(x->val), x); }
template<typename T> ExprPtr<T> tanh(const ExprPtr<T>& x) { return std::make_shared<TanhExpr<T>>(std::tanh(x->val), x); }


//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> exp(const ExprPtr<T>& x) { return std::make_shared<ExpExpr<T>>(std::exp(x->val), x); }
template<typename T> ExprPtr<T> log(const ExprPtr<T>& x) { return std::make_shared<LogExpr<T>>(std::log(x->val), x); }
template<typename T> ExprPtr<T> log10(const ExprPtr<T>& x) { return std::make_shared<Log10Expr<T>>(std::log10(x->val), x); }


//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sqrt(const ExprPtr<T>& x) { return std::make_shared<SqrtExpr<T>>(std::sqrt(x->val), x); }
template<typename T> ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<PowExpr<T>>(std::pow(l->val, r->val), l, r); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const U& l, const ExprPtr<T>& r) { return std::make_shared<PowConstantLeftExpr<T>>(std::pow(l, r->val), constant<T>(l), r); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const ExprPtr<T>& l, const U& r) { return std::make_shared<PowConstantRightExpr<T>>(std::pow(l->val, r), l, constant<T>(r)); }


//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> abs(const ExprPtr<T>& x) { return std::make_shared<AbsExpr<T>>(std::abs(x->val), x); }
template<typename T> ExprPtr<T> abs2(const ExprPtr<T>& x) { return x * x; }
template<typename T> ExprPtr<T> conj(const ExprPtr<T>& x) { return x; }
template<typename T> ExprPtr<T> real(const ExprPtr<T>& x) { return x; }
template<typename T> ExprPtr<T> imag(const ExprPtr<T>& x) { return constant<T>(0.0); }
template<typename T> ExprPtr<T> erf(const ExprPtr<T>& x) { return std::make_shared<ErfExpr<T>>(std::erf(x->val), x); }


//------------------------------------------------------------------------------
// ACTIVATION FUNCTIONS
//------------------------------------------------------------------------------
template <typename T> ExprPtr<T> sigmoid(const ExprPtr<T>& x) { return std::make_shared<SigmoidExpr<T>>(T(1.0) / (T(1.0) + std::exp(-x->val)), x); }
template <typename T> ExprPtr<T> relu(const ExprPtr<T>& x) { return std::make_shared<ReLUExpr<T>>(x->val >= T(0.0) ? x->val : T(0.0), x); }

//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------
template<typename T> bool operator==(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val == r->val; }
template<typename T> bool operator!=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val != r->val; }
template<typename T> bool operator<=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val <= r->val; }
template<typename T> bool operator>=(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val >= r->val; }
template<typename T> bool operator<(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val < r->val; }
template<typename T> bool operator>(const ExprPtr<T>& l, const ExprPtr<T>& r) { return l->val > r->val; }




template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const U& l, const ExprPtr<T>& r) { return l == r->val; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const U& l, const ExprPtr<T>& r) { return l != r->val; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const U& l, const ExprPtr<T>& r) { return l <= r->val; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const U& l, const ExprPtr<T>& r) { return l >= r->val; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const U& l, const ExprPtr<T>& r) { return l < r->val; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const U& l, const ExprPtr<T>& r) { return l > r->val; }

template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const ExprPtr<T>& l, const U& r) { return l->val == r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const ExprPtr<T>& l, const U& r) { return l->val != r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const ExprPtr<T>& l, const U& r) { return l->val <= r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const ExprPtr<T>& l, const U& r) { return l->val >= r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const ExprPtr<T>& l, const U& r) { return l->val < r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const ExprPtr<T>& l, const U& r) { return l->val > r; }

/// The autodiff variable type used for reverse mode automatic differentiation.
template<typename T>
struct Variable
{
    /// The pointer to the expression tree of variable operations
    ExprPtr<T> expr;

    /// Construct a default Variable object
    Variable() : Variable(T(0.0)) {}

    /// Construct a copy of a Variable object
    Variable(const Variable& other) : Variable(other.expr) {}

    /// Construct a var object variable with given int
    // XXX verify
    //var(typename std::enable_if_t<!std::is_same_v<T,int>, int> val) : expr(std::make_shared<ParameterExpr<T>>(static_cast<T>(val))) { }

    /// Construct a Variable object with given arithmetic value
    template<typename U, EnableIf<isArithmetic<U>>...>
    Variable(const U& val) : expr(std::make_shared<IndependentVariableExpr<T>>(val)) {}

    /// Construct a Variable object with given expression
    Variable(const ExprPtr<T>& expr) : expr(expr) {}

    /// Return a pointer to the underlying VariableExpr object in this variable.
    auto __variableExpr() const { return static_cast<VariableExpr<T>*>(expr.get()); }

    /// Return the derivative value stored in this variable.
    auto grad() const { return __variableExpr()->grad; }

    /// Return the derivative expression stored in this variable.
    auto gradx() const { return __variableExpr()->gradx; }

    /// Reeet the derivative value stored in this variable to zero.
    auto seed() { __variableExpr()->grad = 0; }

    /// Reeet the derivative expression stored in this variable to zero expression.
    auto seedx() { __variableExpr()->gradx = constant<T>(0); }

    /// Rewrite and simplify the expression.
    void rewrite() {
      auto new_expr = expr->rewrite();
      if(new_expr) expr = new_expr;
    }

    /// Implicitly convert this Variable object into an expression pointer.
    operator ExprPtr<T>() const { return expr; }

    /// Explicitly convert this Variable object into its underlying arithmetic type.
    explicit operator T() const { return expr->val; }

    /// Assign an arithmetic value to this variable.
    template<typename U, EnableIf<isArithmetic<U>>...>
    auto operator=(const U& val) -> Variable& { *this = Variable(val); return *this; }

    /// Assign an expression to this variable.
    auto operator=(const ExprPtr<T>& x) -> Variable& { *this = Variable(x); return *this; }

	// Assignment operators
    Variable& operator+=(const ExprPtr<T>& x) { *this = Variable(expr + x); return *this; }
    Variable& operator-=(const ExprPtr<T>& x) { *this = Variable(expr - x); return *this; }
    Variable& operator*=(const ExprPtr<T>& x) { *this = Variable(expr * x); return *this; }
    Variable& operator/=(const ExprPtr<T>& x) { *this = Variable(expr / x); return *this; }

	// Assignment operators with arithmetic values
    template<typename U, EnableIf<isArithmetic<U>>...> Variable& operator+=(const U& x) { *this = Variable(expr + x); return *this; }
    template<typename U, EnableIf<isArithmetic<U>>...> Variable& operator-=(const U& x) { *this = Variable(expr - x); return *this; }
    template<typename U, EnableIf<isArithmetic<U>>...> Variable& operator*=(const U& x) { *this = Variable(expr * x); return *this; }
    template<typename U, EnableIf<isArithmetic<U>>...> Variable& operator/=(const U& x) { *this = Variable(expr / x); return *this; }
};

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> bool operator==(const Variable<T>& l, const Variable<T>& r) { return l.expr == r.expr; }
template<typename T> bool operator!=(const Variable<T>& l, const Variable<T>& r) { return l.expr != r.expr; }
template<typename T> bool operator<=(const Variable<T>& l, const Variable<T>& r) { return l.expr <= r.expr; }
template<typename T> bool operator>=(const Variable<T>& l, const Variable<T>& r) { return l.expr >= r.expr; }
template<typename T> bool operator<(const Variable<T>& l, const Variable<T>& r) { return l.expr < r.expr; }
template<typename T> bool operator>(const Variable<T>& l, const Variable<T>& r) { return l.expr > r.expr; }


template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const U& l, const Variable<T>& r) { return l == r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const U& l, const Variable<T>& r) { return l != r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const U& l, const Variable<T>& r) { return l <= r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const U& l, const Variable<T>& r) { return l >= r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const U& l, const Variable<T>& r) { return l < r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const U& l, const Variable<T>& r) { return l > r.expr; }


template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator==(const Variable<T>& l, const U& r) { return l.expr == r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator!=(const Variable<T>& l, const U& r) { return l.expr != r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<=(const Variable<T>& l, const U& r) { return l.expr <= r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>=(const Variable<T>& l, const U& r) { return l.expr >= r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator<(const Variable<T>& l, const U& r) { return l.expr < r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> bool operator>(const Variable<T>& l, const U& r) { return l.expr > r; }


//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> const ExprPtr<T>& operator+(const Variable<T>& r) { return r.expr; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& r) { return -r.expr; }


template<typename T> ExprPtr<T> operator+(const Variable<T>& l, const Variable<T>& r) { return l.expr + r.expr; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& l, const Variable<T>& r) { return l.expr - r.expr; }
template<typename T> ExprPtr<T> operator*(const Variable<T>& l, const Variable<T>& r) { return l.expr * r.expr; }
template<typename T> ExprPtr<T> operator/(const Variable<T>& l, const Variable<T>& r) { return l.expr / r.expr; }


template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& l, const Variable<T>& r) { return l + r.expr; }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& l, const Variable<T>& r) { return l - r.expr; }
template<typename T> ExprPtr<T> operator*(const ExprPtr<T>& l, const Variable<T>& r) { return l * r.expr; }
template<typename T> ExprPtr<T> operator/(const ExprPtr<T>& l, const Variable<T>& r) { return l / r.expr; }


template<typename T> ExprPtr<T> operator+(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr + r; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr - r; }
template<typename T> ExprPtr<T> operator*(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr * r; }
template<typename T> ExprPtr<T> operator/(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr / r; }


template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const U& l, const Variable<T>& r) { return l + r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const U& l, const Variable<T>& r) { return l - r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const U& l, const Variable<T>& r) { return l * r.expr; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const U& l, const Variable<T>& r) { return l / r.expr; }


template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator+(const Variable<T>& l, const U& r) { return l.expr + r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator-(const Variable<T>& l, const U& r) { return l.expr - r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator*(const Variable<T>& l, const U& r) { return l.expr * r; }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> operator/(const Variable<T>& l, const U& r) { return l.expr / r; }


//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sin(const Variable<T>& x) { return sin(x.expr); }
template<typename T> ExprPtr<T> cos(const Variable<T>& x) { return cos(x.expr); }
template<typename T> ExprPtr<T> tan(const Variable<T>& x) { return tan(x.expr); }
template<typename T> ExprPtr<T> asin(const Variable<T>& x) { return asin(x.expr); }
template<typename T> ExprPtr<T> acos(const Variable<T>& x) { return acos(x.expr); }
template<typename T> ExprPtr<T> atan(const Variable<T>& x) { return atan(x.expr); }


//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sinh(const Variable<T>& x) { return sinh(x.expr); }
template<typename T> ExprPtr<T> cosh(const Variable<T>& x) { return cosh(x.expr); }
template<typename T> ExprPtr<T> tanh(const Variable<T>& x) { return tanh(x.expr); }


//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> exp(const Variable<T>& x) { return exp(x.expr); }
template<typename T> ExprPtr<T> log(const Variable<T>& x) { return log(x.expr); }
template<typename T> ExprPtr<T> log10(const Variable<T>& x) { return log10(x.expr); }


//------------------------------------------------------------------------------
// POWER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sqrt(const Variable<T>& x) { return sqrt(x.expr); }
template<typename T> ExprPtr<T> pow(const Variable<T>& l, const Variable<T>& r) { return pow(l.expr, r.expr); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const U& l, const Variable<T>& r) { return pow(l, r.expr); }
template<typename T, typename U, EnableIf<isArithmetic<U>>...> ExprPtr<T> pow(const Variable<T>& l, const U& r) { return pow(l.expr, r); }


//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> abs(const Variable<T>& x) { return abs(x.expr); }
template<typename T> ExprPtr<T> abs2(const Variable<T>& x) { return abs2(x.expr); }
template<typename T> ExprPtr<T> conj(const Variable<T>& x) { return conj(x.expr); }
template<typename T> ExprPtr<T> real(const Variable<T>& x) { return real(x.expr); }
template<typename T> ExprPtr<T> imag(const Variable<T>& x) { return imag(x.expr); }
template<typename T> ExprPtr<T> erf(const Variable<T>& x) { return erf(x.expr); }

//------------------------------------------------------------------------------
// ACTIVATION FUNCTIONS
//------------------------------------------------------------------------------
template <typename T> ExprPtr<T> sigmoid(const Variable<T>& x) { return sigmoid(x.expr); }
template <typename T> ExprPtr<T> relu(const Variable<T>& x) { return relu(x.expr); }

/// Return the value of a scalar.
template<typename U, EnableIf<isArithmetic<U>>...>
auto val(const U& x)
{
    return x;
}

/// Return the value of a variable.
template<typename T>
auto val(const Variable<T>& x)
{
    return val(x.expr->val);
}

/// Return the value of an expression.
template<typename T>
auto val(const ExprPtr<T>& x)
{
    return val(x->val);
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method `derivatives(y, wrt(a, b, c,...)` instead.")]]
auto derivatives(const T& y)
{
    static_assert(!std::is_same_v<T,T>, "Method derivatives(const var&) has been deprecated. Use method derivatives(y, wrt(a, b, c,...) instead.");
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method derivativesx(y, wrt(a, b, c,...) instead.")]]
auto derivativesx(const T& y)
{
    static_assert(!std::is_same_v<T,T>, "Method derivativesx(const var&) has been deprecated. Use method derivativesx(y, wrt(a, b, c,...) instead.");
}

template<typename... Vars>
struct Wrt
{
    std::tuple<Vars...> args;
};

/// The keyword used to denote the variables *with respect to* the derivative is calculated.
template<typename... Args>
auto wrt(Args&&... args)
{
    return Wrt<Args&&...>{ std::forward_as_tuple(std::forward<Args>(args)...) };
}

/// Seed each variable in the **wrt** list.
template<typename... Vars>
auto seed(const Wrt<Vars...>& wrt)
{
    constexpr static auto N = sizeof...(Vars);
    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).seed();
    });
}

/// Seed each variable in the **wrt** list.
template<typename... Vars>
auto seedx(const Wrt<Vars...>& wrt)
{
    constexpr static auto N = sizeof...(Vars);
    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).seedx();
    });
}

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
auto derivatives(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    seed(wrt);
    y.expr->propagate(1.0);

    constexpr static auto N = sizeof...(Vars);
    std::array<T, N> values;
    For<N>([&](auto i) constexpr {
        values[i.index] = std::get<i>(wrt.args).grad();
    });

    return values;
}

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
auto derivativesx(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    seedx(wrt);
    y.expr->propagatex(constant<T>(1.0));

    constexpr static auto N = sizeof...(Vars);
    std::array<Variable<T>, N> values;
    For<N>([&](auto i) constexpr {
        values[i.index] = std::get<i>(wrt.args).gradx();
    });

    return values;
}

/// Output a Variable object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const Variable<T>& x)
{
    out << val(x);
    return out;
}

/// Output an ExprPrt object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const ExprPtr<T>& x)
{
    out << val(x);
    return out;
}

//=====================================================================================================================
//
// HIGHER-ORDER VAR NUMBERS
//
//=====================================================================================================================

template<size_t N, typename T>
struct AuxHigherOrderVariable;

template<typename T>
struct AuxHigherOrderVariable<0, T>
{
    using type = T;
};

template<size_t N, typename T>
struct AuxHigherOrderVariable
{
    using type = Variable<typename AuxHigherOrderVariable<N - 1, T>::type>;
};

template<size_t N, typename T>
using HigherOrderVariable = typename AuxHigherOrderVariable<N, T>::type;

} // namespace reverse

using reverse::wrt;
using reverse::derivatives;
using reverse::Variable;
using reverse::val;

using var = Variable<double>;

} // namespace autodiff
