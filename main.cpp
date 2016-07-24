#include <stdio.h>
#include "bmp.h"

#include "network.h"
#include "training.h"

#include <armadillo>

using namespace arma;

int main()
{
    //Bitmap pic = Bitmap("/home/benko/project/datasets/train-52x52/1/1_0001.bmp");




    Training train1;
    //train1.XOR_training();
    train1.MNIST_training();



  return 1;
}
