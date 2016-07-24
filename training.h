#ifndef TRAINING_H
#define TRAINING_H

#include "neuralnetwork.h"
#include <random>


class Training
{

    inline int random_num(const double, const double);
public:
    Training();

    void XOR_training();
    void MNIST_training();
    void test_case1();
};

#endif // TRAINING_H
