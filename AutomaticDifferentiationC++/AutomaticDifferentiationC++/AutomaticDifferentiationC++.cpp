// AutomaticDifferentiationC++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "./dual.h"
#include <iostream>

int main() {


    // for x = 5 and y = 6
    //the derivative of x**2 * y with respect to x is
    // y * 2 * x which is 6 * 5 * 2 is 60

    // form of (val, der), wehre der = 1 means we are going to evaluate the
    //the derivative with respect to this variable
    Dual x(5, 1);
    Dual y(6); //since der = 0, we will not evaluate the derivtive with respect to y

    Dual f = pow(x, 2) * y;

    std::cout << f.getDerivative() << std::endl;
    return 0;
}

