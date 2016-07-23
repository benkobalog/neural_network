#include "training.h"
#include "network.h"



Training::Training()
{

}


void Training::train_neural_network()
{
    // for read training data
    std::vector<uint> fc_topology {2,3,2};
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

    double learning_rate = 1;
    for (int var = 0; var < 20000; ++var) {
        const double lower_bound = 0;
        const double upper_bound = 4;
        std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
        std::random_device rand_dev;
        std::mt19937 rand_engine(rand_dev());
        double rnd_d = unif(rand_engine);
        int rnd = (int)rnd_d;


        std::cout << "Pass: " << var;
        std::cout << " Inputs: " << x[rnd](0, 0) << " " << x[rnd](1, 0) << std::endl;
        std::cout << "Target: " << y[rnd](0, 0) << std::endl;


        nn2.stochastic_gradient_descent(x[rnd], y[rnd]);
        //nn2.mini_batch(x[1], y[1]);;

        std::cout << std::endl;

        uint batch_size = 8;
        if (var % batch_size == 0)
        {
            nn2.update_weights(batch_size, learning_rate);
            cout << "learning rate===================== " << learning_rate << endl;
        }
        cout << endl ;
    }

    nn2.print_weights();

        // pass training data
        // call mini_batch

   // evaluation?
}


void Training::test_case1()
{
    // for read training data
    std::vector<uint> fc_topology {2,3,1};
    NeuralNetwork nn2(fc_topology);

    // XOR
    mat x1 = { 0, 0 }; mat y1 =  { 0 };
    mat x2 = { 0, 1 }; mat y2 =  { 1 };
    mat x3 = { 1, 0 }; mat y3 =  { 1 };
    mat x4 = { 1, 1 }; mat y4 =  { 1 };

    nn2.mini_batch(x1, y1);
    nn2.mini_batch(x2, y2);
    nn2.mini_batch(x3, y3);
    nn2.mini_batch(x4, y4);
        // pass training data
        // call mini_batch

   // evaluation?
}
