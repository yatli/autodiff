// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

int main()
{
    var<double> x = 0.5;                              // the input variable x
    var<double> u = sin(x) * cos(x);                  // the output variable u

    DerivativesX<double> dud = derivativesx(u);       // evaluate the first order derivatives of u

    var<double> dudx = dud(x);                        // extract the first order derivative du/dx of type var, not double!

    DerivativesX<double> d2udxd = derivativesx(dudx); // evaluate the second order derivatives of du/dx

    var<double> d2udxdx = d2udxd(x);                  // extract the second order derivative d2u/dxdx of type var, not double!

    cout << "u = " << u << endl;              // print the evaluated output variable u
    cout << "du/dx = " << dudx << endl;       // print the evaluated first order derivative du/dx
    cout << "d2u/dx2 = " << d2udxdx << endl;  // print the evaluated second order derivative d2u/dxdx
}
