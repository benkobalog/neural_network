#include <stdio.h>
#include "bmp.h"

#include "network.h"
#include "training.h"

int main()
{
    Training train1;
    //train1.XOR_training();
    train1.MNIST_training();
  return 1;
}
