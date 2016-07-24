#include <stdio.h>
#include "bmp.h"

#include "network.h"
#include "training.h"

#include <armadillo>

using namespace arma;

int main()
{
    //Bitmap pic = Bitmap("/home/benko/project/datasets/train-52x52/1/1_0001.bmp");

    //pic.print_pixels();
    //NeuralNetwork nn1(3,4,3);
   /* std::vector<uint> fc_topology {2,3,2};
    NeuralNetwork nn2(fc_topology);


    mat input = { 1, 1 };
    nn2.feedforward(input);
*/
    Training train1;
    train1.XOR_training();



  return 1;
}
