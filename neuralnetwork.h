#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <algorithm>
#include <random>
#include <iostream>
#include <armadillo>

using namespace arma;

typedef unsigned int uint;


class Layer
{
    mat W;
    uint nr_prev_layer_neurons;
    // Activation units

    void ReLu();
    double sigmoid();
    double softMax();
    double rnd();
    void init_weights();
    void set_activation();

public:
    mat neurons;

    Layer(const uint, const uint);

    mat forward_prop(const mat&);
    void print_neurons();

};



class NeuralNetwork
{

    // Variables
    std::vector<uint> num_of_neurons;
    std::vector<Layer> network;

public:
    NeuralNetwork(const std::vector<uint> &);
    void forward_prop(const mat&);


    void back_prop();

};




// Neuron, Activation



// BackProp

// Gradient



#endif // NEURALNETWORK_H
