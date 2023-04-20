#pragma once
#ifndef DUAL
#define DUAL

#include <iostream>
#include <cmath>

class Dual {
private:
    double val;
    double der;

public:
    Dual();
    Dual(double var);
    Dual(double var, double der);

    // getters and setters
    double getDerivative() const;
    void setDerivative(double der);

    //overloaded operators
    friend Dual operator+(const Dual& u, const Dual& v);
    friend Dual operator-(const Dual& u, const Dual& v);
    friend Dual operator*(const Dual& u, const Dual& v);
    friend Dual operator/(const Dual& u, const Dual& v);

    //overloaded printing
    friend std::ostream& operator<<(std::ostream& os, const Dual& d);

    //functions whose derivatives we can take
    // if need be, it's easy to add new functions and expand the scope of
    //ability of the program
    friend Dual sin(Dual d);
    friend Dual cos(Dual d);
    friend Dual exp(Dual d);
    friend Dual log(Dual d);
    friend Dual abs(Dual d);
    friend Dual pow(Dual d, double exponent);




};
#endif