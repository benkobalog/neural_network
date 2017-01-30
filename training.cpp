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

        double error = 0.0;
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

    double** imgs = new double*[nr_images];
    //convert images to 0-1 range from 0-255
    for(int i = 0; i < nr_images; i++) {
        imgs[i] = new double[image_size];
        for (int col = 0; col < 28; ++col) {
            for (int row = 0; row < 28; ++row) {
                uint index = col * 28 + row;
                imgs[i][index] = (double)images[i][index] / 256;
            }
        }
    }

    std::vector<uint> fc_topology {28*28, 500, 10};
    network::Network nn2(fc_topology);

    // loop through data
    // parameters
    uint batch_size = 10;
    double learning_rate = 1.5;
    double lambda = 0.001;
    double eta = 0.99;
    mat x,y;

    double training_acc = 0;

    double last_error = 100;
    double sum_error = 0;
    for(int i = 0; i < nr_images; i++)
    {
        //cout << (int)labels[i] << " " << endl;
        y = zeros(10,1);
        x = zeros(28 * 28, 1);
        y((int)labels[i], 0) = 1;


        for (int col = 0; col < 28; ++col)
        {
            for (int row = 0; row < 28; ++row)
            {
                uint index = col * 28 + row;
                x(index, 0) = imgs[i][index];
            }
        }

        network::Training_results results = nn2.stochastic_gradient_descent(x, y);
        sum_error += results.error;

        if( results.prediction == (int)labels[i] )
        {
            training_acc++;
        }

        if (i % batch_size == 0)
        {
            if (last_error * eta > sum_error){
                last_error = sum_error;
                learning_rate *= .9;
                //eta *= 1.01;
            }
            nn2.update_weights(batch_size, learning_rate, lambda);
            cout << "learning rate===================== " << learning_rate << " sum " << sum_error << " last "<< last_error << endl;
            sum_error = 0.0;
        }
        //cout << endl ;
    }

    training_acc /= nr_images;
    //nn2.print_weights();


    // Test phase ==============================================
    /*delete images;
    delete imgs;
    delete labels;*/
    // TODO Delete images and labels, now memory leak

    cout << "Testing................\n";

    uchar** t_images = read_mnist_images("/home/benko/project/datasets/mnist/t10k-images.idx3-ubyte",nr_images, image_size);
    uchar* test_labels = read_mnist_labels("/home/benko/project/datasets/mnist/t10k-labels.idx1-ubyte", nr_labels);

    double** test_images = new double*[nr_images];
    mat debug ;
    //convert images to 0-1 range from 0-255
    for(int i = 0; i < nr_images; i++)
    {
        test_images[i] = new double[image_size];
        for (int col = 0; col < 28; ++col)
        {
            for (int row = 0; row < 28; ++row)
            {
                uint index = col * 28 + row;
                test_images[i][index] = (double)t_images[i][index] / 256;
            }
        }
    }

    // load test images
    double accuracy = 0;
    int true_positive = 0;

    for(int i = 0; i < nr_images; i++)
    {
        //cout << (int)labels[i] << " " << endl;
        y = zeros(10,1);
        x = zeros(28 * 28, 1);
        debug = zeros(28, 28);
        y((int)test_labels[i], 0) = 1;

        for (int col = 0; col < 28; ++col)
        {
            for (int row = 0; row < 28; ++row)
            {
                uint index = col * 28 + row;
                x(index, 0) = test_images[i][index];
                debug(col, row) = test_images[i][index];
            }
        }

        int prediction = nn2.predict(x, y);

        cout << "prediction: " << prediction << " target: " << (int)test_labels[i] << endl;
        //debug.print("debug");

        if( prediction == (int)test_labels[i] )
        {
            true_positive++;
        }

        cout << endl ;
    }

    accuracy = (double)(true_positive) / (double)nr_labels;
    cout << "Test accuracy     = "<< accuracy << " : "<< true_positive << "/" << nr_labels<< endl;
    cout << "Training accuracy = " << training_acc<< endl;
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
