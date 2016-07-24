#include "network.h"

using namespace network;


Network::Network(const std::vector<uint> &nr_neurons)
{

    // Input layer
    layers.push_back(Layer(nr_neurons[0]));

    for (uint i = 1; i < nr_neurons.size(); ++i) {
        layers.push_back(Layer(layers[i-1], nr_neurons[i]));
    }
    // good weights for the xor problem
    /*layers[1].W(0,0) = 20; layers[1].W(0,1) = 20;
    layers[1].W(1,0) = -20; layers[1].W(1,1) = -20;

    layers[1].bias(0,0) = -20;
    layers[1].bias(1,0) = 30;

    layers[2].W(0,0) = 20; layers[2].W(0,1) = 20;
    layers[2].bias(0,0) = -30;*/

   /* layers[1].W(0,0) = -1; layers[1].W(0,1) = 1;
    layers[1].W(1,0) = 1; layers[1].W(1,1) = -1;

    layers[1].bias(0,0) = -0;
    layers[1].bias(1,0) = 0;

    layers[2].W(0,0) = -1; layers[2].W(0,1) =  -1;
    layers[2].W(1,0) = 1; layers[2].W(1,1) =  1;
    layers[2].bias(0,0) = 1;
    layers[2].bias(1,0) = 0;*/

}

void Network::feedforward(const mat& input)
{
    cout << "feedforward\n";
    layers[0].A = input;
    for (uint i = 1; i < layers.size(); ++i) {
        layers[i].Z = layers[i].W * layers[i-1].A + layers[i].bias;
        if (i == layers.size())
        {
            Activations::set_activation(layers[i].Z, layers[i].A, true);
        }
        else
        {
            Activations::set_activation(layers[i].Z, layers[i].A, false);
        }
    }

    layers[layers.size() - 1].A.print("output: ");
}

void Network::stochastic_gradient_descent(const mat &x, const mat& y)
{
    feedforward(x);

    int i_output_layer = layers.size() - 1;

    mat error = 0.5 * (layers[i_output_layer].A - y) % (layers[i_output_layer].A - y);
    double er = 0;
    for (uint var = 0; var < error.size(); ++var) {
        er += error(var,0);
    }
    cout << "error: " << er << endl;

    // calc output error
    mat grad_cost = layers[i_output_layer].A - y;
    layers[i_output_layer].delta = grad_cost % activation_derivate(layers[i_output_layer].Z, true);

    layers[i_output_layer].nabla_w = layers[i_output_layer].nabla_w + (layers[i_output_layer].delta * layers[i_output_layer-1].A.t());
    layers[i_output_layer].nabla_b = layers[i_output_layer].nabla_b + (layers[i_output_layer].delta);

    //back propagate error
    for (uint i = i_output_layer - 1; i > 0; --i) {
        layers[i].delta = (layers[i + 1].W.t() * layers[i + 1].delta) % activation_derivate(layers[i].Z, false);

        layers[i].nabla_w = layers[i].nabla_w + (layers[i].delta * layers[i-1].A.t());
        //layers[i].nabla_b = layers[i].nabla_b + (layers[i].bias - layers[i].A);
        layers[i].nabla_b = layers[i].nabla_b + (layers[i].delta);
    }

    //layers[i_output_layer].W.print("W");


}

void Network::update_weights(const uint batch_size, double & learning_rate, double & lambda)
{
    //update weights


    // layers[0] is just the input now
    for (uint i = 1; i < layers.size(); ++i)
    {

        layers[i].W     = layers[i].W    - learning_rate/batch_size * layers[i].nabla_w;
        layers[i].bias  = layers[i].bias - (double)(learning_rate/batch_size) * layers[i].nabla_b;

        // weight decay
        layers[i].W    = layers[i].W    - learning_rate * lambda * layers[i].W;
        //layers[i].bias = layers[i].bias - learning_rate * lambda * layers[i].bias;

        layers[i].nabla_w = zeros(layers[i].W.n_rows,    layers[i].W.n_cols);
        layers[i].nabla_b = zeros(layers[i].bias.n_rows, layers[i].bias.n_cols);

    }

    if (learning_rate > 1 )
    {
        learning_rate *= .9991;
    }
    else if ( learning_rate > .1)
    {
        learning_rate *= .9999;
    }

}

mat Network::activation_derivate(const mat& matrix, const bool is_sigmoid)
{
    // Output layer
    mat ret = zeros(matrix.n_rows, 1);

    for (uint var = 0; var < matrix.n_rows; ++var)
    {
        if (is_sigmoid)
        {
            ret(var, 0) = Activations::sigmoid_derivative(matrix(var,0));
        }
        else
        {
            ret(var, 0) = Activations::ReLu_derivative(matrix(var,0));
        }
    }
    return ret;
}


void Network::print_weights()
{
    for (uint i = 1; i < layers.size(); ++i)
    {

       // layers[i].nabla_w.print("nab W");
       // layers[i].nabla_b.print("nab B");
        layers[i].W.print("W");
        layers[i].bias.print("B");

    }
}

//===================================
//=========== Layer =================
//===================================



Layer::Layer(const Layer& prev_layer, const uint nr_neurons)  : Z(zeros<mat>(nr_neurons, 1)), A(zeros<mat>(nr_neurons, 1))
{
    init_weights_and_bias(prev_layer, nr_neurons);
}


Layer::Layer(const uint nr_neurons) : Z(zeros<mat>(nr_neurons, 1)), A(zeros<mat>(nr_neurons, 1))
{
    // Input Layer

}

double Layer::normal_dist_number(double mean, double variance)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, variance);
    return d(gen);
}

void Layer::xavier_init(double nr_neuron, mat& matrix)
{
    double mean = 0.0;
    // TODO this is possibly the output instead of the current neuron number
    double variance = 1.0 / nr_neuron;

    for (uint col = 0; col < matrix.n_cols; ++col) {
        for (uint row = 0; row < matrix.n_rows; ++row) {
            matrix(row, col) = normal_dist_number(mean, variance);
        }
    }
}

void Layer::init_weights_and_bias(const Layer& prev_layer, const uint nr_neurons)
{
    std::cout << "Initialising layer\n";
    arma_rng::set_seed_random();
    //uint nr_neurons = neurons.n_cols;

    cout << "tocs: " << nr_neurons << " " << prev_layer.Z.n_rows << endl;

    W.zeros(nr_neurons, prev_layer.Z.n_rows);
    xavier_init(nr_neurons, W);
    W.print("w: ");
    nabla_w = zeros(W.n_rows, W.n_cols);
    if (prev_layer.Z.n_rows > 0 )
    {
        bias.zeros(nr_neurons, 1);
        xavier_init(nr_neurons, bias);
        bias.print("bias: ");

        nabla_b = zeros(bias.n_rows, bias.n_cols);
    }
}


//====================================================
//====================Activations=====================
//====================================================


void Activations::set_activation(const mat &Z , mat &A, const bool is_sigmoid)
{
    for (uint var = 0; var < A.n_rows; ++var)
    {
        if (is_sigmoid)
        {
            A(var, 0) = Activations::sigmoid(Z(var, 0));
        }
        else
        {
            A(var, 0) = Activations::ReLu(Z(var, 0));
        }
    }
}


double Activations::sigmoid_derivative(double input)
{
    double tmp = sigmoid(input);
    return tmp * (1 - tmp);
}

double Activations::ReLu(double input)
{
    if (0.0 > input)
    {
        input = 0.0;
    }
    return input;
}

double Activations::ReLu_derivative(double input)
{
    if (0.0 >= input)
    {
        input = 0.0;
    }
    else
    {
        input = 1.0;
    }
    return input;
}

double Activations::sigmoid(double input)
{
    //neurons(0, index) = 1.0 / (1.0 + std::exp(-neurons(0, index)));
    return 1.0 / (1.0 + std::exp(-input));
}
