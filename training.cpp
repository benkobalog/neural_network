#include "training.h"
#include "network.h"

#include "mnist.h"



Training::Training()
{

}


void Training::XOR_training()
{
    // for read training data
    std::vector<uint> fc_topology {2,2,2};
    //NeuralNetwork nn2(fc_topology);

    network::Network nn2(fc_topology);


    // XOR
    std::vector<mat> x;
    std::vector<mat> y;

    mat tmp;
    tmp = zeros(2,1); tmp(0,0) = 0; tmp(1,0) = 0;
    x.push_back(tmp);
    tmp = zeros(2,1); tmp(0,0) = 0; tmp(1,0) = 1;
    x.push_back(tmp);
    tmp = zeros(2,1); tmp(0,0) = 1; tmp(1,0) = 0;
    x.push_back(tmp);
    tmp = zeros(2,1); tmp(0,0) = 1; tmp(1,0) = 1;
    x.push_back(tmp);

    /*
    x[0] = zeros(2,1); x[0](0,0) = 1; x[0](1,0) = 1;
    x[1] = zeros(2,1); x[1](0,0) = 0; x[1](1,0) = 0;
    x[2] = zeros(2,1); x[2](0,0) = 0; x[2](1,0) = 1;
    x[3] = zeros(2,1); x[3](0,0) = 1; x[3](1,0) = 0;
*/
    tmp = zeros(2,1); tmp(0,0) = 1; tmp(1,0) = 0; // 0
    y.push_back(tmp);
    tmp = zeros(2,1); tmp(0,0) = 0; tmp(1,0) = 1; // 1
    y.push_back(tmp);
    y.push_back(tmp);
    tmp = zeros(2,1); tmp(0,0) = 1; tmp(1,0) = 0; // 0
    y.push_back(tmp);

  /*  y.push_back({0});
    y.push_back({1});
    y.push_back({1});
    y.push_back({0});  */

    double learning_rate = .1;
    double lambda = .0001;
    for (int var = 0; var < 20000; ++var) {

        int rnd = random_num(0,4);

        std::cout << "Pass: " << var;
        std::cout << " Inputs: " << x[rnd](0, 0) << " " << x[rnd](1, 0) << std::endl;
        std::cout << "Target: " << y[rnd](0, 0) << std::endl;


        nn2.stochastic_gradient_descent(x[rnd], y[rnd]);
        //nn2.mini_batch(x[1], y[1]);;

        std::cout << std::endl;

        uint batch_size = 8;
        if (var % batch_size == 0)
        {
            nn2.update_weights(batch_size, learning_rate, lambda);
            cout << "learning rate===================== " << learning_rate << endl;
        }
        cout << endl ;
    }

    nn2.print_weights();
}

void Training::MNIST_training()
{
    // Read dataset
    int nr_images, image_size, nr_labels;

    uchar** images = read_mnist_images("/home/benko/project/datasets/mnist/train-images.idx3-ubyte",nr_images, image_size);
    uchar*  labels = read_mnist_labels("/home/benko/project/datasets/mnist/train-labels.idx1-ubyte", nr_labels);

    for(int i = 0; i < nr_images; i++) {
        cout << (int)labels[i] << " " << endl;

        for (int col = 0; col < 28; ++col) {
            for (int row = 0; row < 28; ++row) {
                uint index = col * 28 + row;
                cout << (int)images[i][index] << " ";
            }
            cout << endl;
        }
        cout << "=======================================\n\n\n";
    }


    // loop through data

    // update weights
}


void Training::test_case1()
{

}


int Training::random_num(const double lower_bound , const double upper_bound)
{
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    double rnd_d = unif(rand_engine);
    return (int)rnd_d;
}
