// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff include
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
template<typename T>
var<T> f(const VectorXvar<T>& x)
{
    return sqrt(x.cwiseProduct(x).sum()); // sqrt(sum([x(i) * x(i) for i = 1:5]))
}

int main()
{
    VectorXvar<double> x(5);               // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    var<double> y = f(x);                          // the output variable y

    VectorXvar<double> dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    cout << "y = " << y << endl;           // print the evaluated output y
    cout << "dy/dx = \n" << dydx << endl;  // print the evaluated gradient vector dy/dx
}
