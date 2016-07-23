#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <armadillo>

using namespace arma;
using namespace std;

namespace network
{

    struct Activations
    {
        // Activation units
        static inline double ReLu(double);
        static inline double ReLu_derivative(double);
        static inline double sigmoid(double);
        static inline double sigmoid_derivative(double input);

        static inline void set_activation(const mat&, mat&, const bool);

    };

    class Layer
    {
        mat bias;
        mat W;
        mat Z;
        mat A;
        mat delta;
        mat nabla_w;
        mat nabla_b;

        void init_weights_and_bias(const Layer&, const uint);
        void xavier_init(double nr_neuron, mat& matrix);
        double normal_dist_number(double mean, double variance);


        Layer(const Layer&, const uint);
        Layer(const uint);


        friend class Network;
    };




    class Network
    {
        std::vector<Layer> layers;
    public:
        void feedforward(const mat& input);
        void stochastic_gradient_descent(const mat&, const mat&);
        void update_weights(const uint, double&);
        mat activation_derivate(const mat&, const bool is_sigmoid);
        void print_weights();
        Network(const std::vector<uint>&);
    };


}
#endif // NETWORK_H
