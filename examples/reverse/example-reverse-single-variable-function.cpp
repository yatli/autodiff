// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The single-var<double>iable function for which derivatives are needed
var<double> f(var<double> x)
{
    return 1.0 + x + x*x + 1.0/x + log(x);
}

int main()
{
    var<double> x = 2.0;                         // the input var<double>iable x
    var<double> u = f(x);                        // the output var<double>iable u

    Derivatives<double> dud = derivatives(u);    // evaluate all derivatives of u

    var<double> dudx = dud(x);                   // extract the derivative du/dx

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
}
