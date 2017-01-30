#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <iostream>

typedef unsigned int uint;
using namespace std;

struct mat
{
    unsigned int n_cols;
    unsigned int n_rows;


    // *
    mat operator* (const mat&);
    mat operator* (const double&);

    // +
    mat operator+ (const mat&);

    // %
    mat operator% (const mat&);

    // -
    mat operator- (const mat&);


    mat t();


    // ()
    double& operator() (const uint, const uint);
    double operator() (const uint, const uint) const;

    // =
    mat operator= (const mat&);

    void print(string);

    vector<vector<double>> values;
private:


};

// zero init
extern mat zeros(uint, uint);
//template <class> static mat zeros(uint, uint);

#endif // MATRIX_H
