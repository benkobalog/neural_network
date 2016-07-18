#include "neuralnetwork.h"

//======= NeuralNetwork =================
NeuralNetwork::NeuralNetwork(const std::vector<uint> &neurons) : num_of_neurons(std::move(neurons))
{
    uint nr_prev_layer_neurons = 0;
    for ( auto const &num : num_of_neurons )
    {
        network.push_back(Layer(num, nr_prev_layer_neurons));
        nr_prev_layer_neurons = num;
    }


}


void NeuralNetwork::forward_prop(const mat& input)
{
    mat output = input;
    // iterate through the layers
    for (uint var = 1; var < network.size(); ++var)
    {
        //pass the inputs/outputs
        output = network[var].forward_prop(output);
        //call layer.forward_prop
        //network[var].neurons = output;
        output.print("outputs:");
    }


}

void NeuralNetwork::mini_batch()
{
    // set input

    // feedforward

    // output error

    // backprop error

    // gradient descent (update weights)
}

//=========== Layer =================

void Layer::ReLu(int index)
{
    if (0.0 > neurons(0,index))
    {
        neurons(0, index) = 0.0;
    }
}

void Layer::sigmoid(int index)
{
    neurons(0, index) = 1.0 / (1.0 + std::exp(-neurons(0, index)));
}


Layer::Layer(uint nr_neurons, uint nr_prev_layer_neurons)
    : nr_prev_layer_neurons(nr_prev_layer_neurons),
      neurons(ones<mat>(1,nr_neurons))
{
    //print_neurons();
    init_weights_and_bias();
}

void Layer::init_weights_and_bias()
{
    std::cout << "Initialising layer\n";
    arma_rng::set_seed_random();
    W.randu(nr_prev_layer_neurons, neurons.size());
    W.print("w: ");
    if (nr_prev_layer_neurons > 0 )
    {
        bias.randu(1, neurons.size());
        bias.print("bias: ");
    }
}

void Layer::print_neurons()
{
    for (uint var = 0; var < neurons.size(); ++var)
    {
        std::cout << neurons[var] << " ";
    }
    std::cout << std::endl;
}

void Layer::set_activation()
{
    for (uint var = 0; var < neurons.n_cols; ++var) {
        //ReLu(var);
        sigmoid(var);
    }
}

mat Layer::forward_prop(const mat &input)
{
    // TODO this without copy contructor
    std::cout << "FeedForward\n";
    neurons = input * W + bias;
    set_activation();
    return neurons;
}

double Layer::rnd()
{
    const double lower_bound = 0;
    const double upper_bound = 1;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

    std::random_device rand_dev;          // Use random_device to get a random seed.

    std::mt19937 rand_engine(rand_dev()); // mt19937 is a good pseudo-random number
                                          // generator.

    double x = unif(rand_engine);
    std::cout << x << std::endl;
    return x;
}
