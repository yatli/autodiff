// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The multi-var<double>iable function for which derivatives are needed
var<double> f(var<double> x, var<double> y, var<double> z)
{
    return 1.0 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var<double> x = 1.0;                         // the input var<double>iable x
    var<double> y = 2.0;                         // the input var<double>iable y
    var<double> z = 3.0;                         // the input var<double>iable z
    var<double> u = f(x, y, z);                  // the output var<double>iable u

    Derivatives<double> dud = derivatives(u);    // evaluate all derivatives of u

    var<double> dudx = dud(x);                   // extract the derivative du/dx
    var<double> dudy = dud(y);                   // extract the derivative du/dy
    var<double> dudz = dud(z);                   // extract the derivative du/dz

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/dy = " << dudy << endl;  // print the evaluated derivative du/dy
    cout << "du/dz = " << dudz << endl;  // print the evaluated derivative du/dz
}
