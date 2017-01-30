#include "matrix.h"


mat mat::t()
{
    mat ret;
    ret = zeros(n_cols,n_rows);

    for (uint col = 0; col < n_rows; ++col) {
        for (uint row = 0; row < n_cols; ++row) {
            ret(row,col) = values[col][row];
        }
    }
    return ret;
}

mat mat::operator *(const mat& other)
{
    mat ret;

    if ( this->n_cols != other.n_rows ) {
        cout << "Dimensions don't' match for matrix multiply "
             << this->n_cols << " != "  << other.n_rows
             << endl;
        return ret;
    }

    ret = zeros(this->n_rows, other.n_cols);


    for (uint row = 0; row < ret.n_rows; ++row) {
        for (uint col = 0; col < ret.n_cols; ++col) {
            // Calculate one field of the new matrix
            for (uint inner = 0; inner < this->n_cols; ++inner) {
                ret.values[row][col] += this->values[col][inner] * other.values[inner][row];
            }
        }
    }

    return ret;
}

mat mat::operator *(const double& A)
{
    mat ret;

    ret = zeros(this->n_rows, this->n_cols);


    for (uint row = 0; row < this->n_rows; ++row) {
        for (uint col = 0; col < this->n_cols; ++col) {
            ret.values[row][col] = this->values[row][col] * A;
        }
    }

    return ret;
}


mat mat::operator %(const mat& other)
{
    // Hadamard product
    mat ret;
    if ( this->n_cols != other.n_cols || this->n_rows != other.n_rows ) {
        cout << "Dimensions don't' match "
             << this->n_cols << " != "  << other.n_cols
             << " or "
             << this->n_rows << " != "  << other.n_rows
             << endl;
        return ret;
    }

    ret  = zeros(this->n_rows, this->n_cols);


    for (uint row = 0; row < this->n_rows; ++row) {
        for (uint col = 0; col < this->n_cols; ++col) {
            ret.values[row][col] = this->values[row][col] * other.values[row][col];
        }
    }

    return ret;
}


mat mat::operator +(const mat& other)
{
    mat ret;
    if ( this->n_cols != other.n_cols || this->n_rows != other.n_rows ) {
        cout << "Dimensions don't' match "
             << this->n_cols << " != "  << other.n_cols
             << " or "
             << this->n_rows << " != "  << other.n_rows
             << endl;
        return ret;
    }


    ret = zeros(this->n_rows, this->n_cols);

    for (uint row = 0; row < this->n_rows; ++row) {
        for (uint col = 0; col < this->n_cols; ++col) {
            ret.values[row][col] = this->values[row][col] + other.values[row][col];
        }
    }

    return ret;
}


mat mat::operator -(const mat& other)
{
    mat ret;
    if ( this->n_cols != other.n_cols || this->n_rows != other.n_rows ) {
        cout << "Dimensions don't' match "
             << this->n_cols << " != "  << other.n_cols
             << " or "
             << this->n_rows << " != "  << other.n_rows
             << endl;
        return ret;
    }

    ret = zeros(this->n_rows, this->n_cols);


    for (uint row = 0; row < this->n_rows; ++row) {
        for (uint col = 0; col < this->n_cols; ++col) {
            ret.values[row][col] = this->values[row][col] - other.values[row][col];
        }
    }

    return ret;
}





double& mat::operator() (const uint x , const uint y)
{
    return values[x][y];
}

double mat::operator() (const uint x , const uint y) const
{
    return values[x][y];
}


mat mat::operator= (const mat& other)
{

    this->n_cols = other.n_cols;
    this->n_rows = other.n_rows;

    this->values = other.values;
    return *this;
}




// TODO these functions have to be changed, now they are ugly.
mat zeros(uint rows, uint cols)
{
    mat ret;
    ret.n_cols = cols; ret.n_rows = rows;
    ret.values.resize(ret.n_rows);
    for (uint i = 0; i < ret.values.size(); ++i) {
        ret.values[i].resize(ret.n_cols);
    }

    for (uint col = 0; col < ret.n_cols; ++col) {
        for (uint row = 0; row < ret.n_rows; ++row) {
            ret(row, col) = 0;
        }
    }
    return ret;
}
/*
template<class T>
mat zeros(uint rows, uint cols)
{
    mat ret;
    ret.n_cols = cols; ret.n_rows = rows;
    ret.values.resize(ret.n_rows);
    for (uint i = 0; i < ret.values.size(); ++i) {
        ret.values[i].resize(ret.n_cols);
    }

    for (uint col = 0; col < ret.n_cols; ++col) {
        for (uint row = 0; row < ret.n_rows; ++row) {
            ret(row, col)  = 1;
        }
    }
    return ret;
}

*/

void mat::print(string s)
{
    cout << s << endl;
    for (uint row = 0; row < n_rows; ++row) {
        for (uint col = 0; col < n_cols; ++col) {
            cout << values[row][col] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
