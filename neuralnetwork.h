#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <armadillo>

using namespace arma;
using namespace std;

typedef unsigned int uint;



struct Activations
{
    // Activation units
    static inline double ReLu(double);
    static inline double ReLu_derivative(double);
    static inline double sigmoid(double);
    static inline double sigmoid_derivative(double input);

};

class Layer
{
    uint nr_prev_layer_neurons;

    double rnd();
    void init_weights_and_bias();
    void set_activation();
    double normal_dist_number(double, double);
    void xavier_init(double, mat&);

public:
    mat bias;
    mat W;
    mat neurons;
    mat delta;

    Layer(const uint, const uint);

    void feedforward(const mat&);
    void print_neurons();

};



class NeuralNetwork
{

    // Variables
    std::vector<uint> num_of_neurons;
    std::vector<Layer> layers;

    mat grad_cost(mat);
    mat activation_derivate(int i_layer);

public:

    NeuralNetwork(const std::vector<uint> &);
    void feedforward(const mat&);
    void mini_batch(mat, mat);
    void update_weights(mat);

    void back_prop();

};




// Neuron, Activation



// BackProp

// Gradient



#endif // NEURALNETWORK_H
