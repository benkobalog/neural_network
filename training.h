#ifndef TRAINING_H
#define TRAINING_H

#include <random>
typedef unsigned char uchar;

//#include "matrix.h"

class Training
{
    inline int random_num(const double, const double);
public:
    Training();

    void XOR_training();
    void MNIST_training();
    void test_case1();
    double ** read_training_image(int nr_images, int image_size, uchar** images);
    double ** read_test_images(int image_size, int nr_images, uchar** t_images);
};

#endif // TRAINING_H
