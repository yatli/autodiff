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
#include "reverse.hpp"
#include "Eigen/src/Core/util/XprHelper.h"
#include "Eigen/src/Core/Matrix.h"

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF VAR
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<typename T> struct NumTraits<autodiff::var<T>> : NumTraits<T> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef autodiff::var<T> Real;
  typedef autodiff::var<T> NonInteger;
  typedef autodiff::var<T> Nested;
  enum
    {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 1,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 3,
      MulCost = 3
    };
};

template<typename BinOp, typename T>
struct ScalarBinaryOpTraits<autodiff::var<T>, T, BinOp>
{
  typedef autodiff::var<T> ReturnType;
};

template<typename BinOp, typename T>
struct ScalarBinaryOpTraits<T, autodiff::var<T>, BinOp>
{
    typedef autodiff::var<T> ReturnType;
};

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
template<typename T> \
using Matrix##SizeSuffix##TypeSuffix = Matrix<Type, Size, Size, 0, Size, Size>;  \
template<typename T> \
using Vector##SizeSuffix##TypeSuffix = Matrix<Type, Size, 1, 0, Size, 1>;  \
template<typename T> \
using RowVector##SizeSuffix##TypeSuffix = Matrix<Type, 1, Size, 1, 1, Size>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
template<typename T> \
using Matrix##Size##X##TypeSuffix = Matrix<Type, Size, -1, 0, Size, -1> ;\
template<typename T> \
using Matrix##X##Size##TypeSuffix = Matrix<Type, -1, Size, 0, -1, Size> ;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, -1, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::var<T>, var)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // namespace Eigen

namespace autodiff {

/// Return the gradient vector of variable y with respect to variables x.
template<typename vars, typename T>
Eigen::RowVectorXvar<T> gradient(const var<T>& y, const vars& x)
{
    const auto n = x.size();
    Eigen::RowVectorXvar<T> dydx(n);
    Derivatives<T> dyd = derivatives(y);
    for(auto i = 0; i < n; ++i)
        dydx[i] = dyd(x[i]);
    return dydx;
}

/// Return the Hessian matrix of variable y with respect to variables x.
template<typename vars, typename T>
Eigen::MatrixXvar<T> hessian(const var<T>& y, const vars& x)
{
    const auto n = x.size();
    Eigen::MatrixXvar<T> mat(n, n);
    DerivativesX<T> dyd = derivativesx(y);
    for(auto i = 0; i < n; ++i)
    {
        Derivatives<T> d2yd = derivatives(dyd(x[i]));
        for(auto j = i; j < n; ++j) {
            mat(i, j) = mat(j, i) = d2yd(x(j));
        }
    }
    return mat;
}

} // namespace autodiff
