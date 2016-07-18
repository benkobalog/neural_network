#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <armadillo>

using namespace arma;

typedef unsigned int uint;


class Layer
{
    mat W;
    mat bias;
    uint nr_prev_layer_neurons;

    // Activation units
    inline double ReLu(double);
    inline double sigmoid(double);
    //inline void softMax();

    inline double sigmoid_derivate(double input);

    //Activation
    double rnd();
    void init_weights_and_bias();
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
    void mini_batch();

    void back_prop();

};




// Neuron, Activation



// BackProp

// Gradient



#endif // NEURALNETWORK_H
