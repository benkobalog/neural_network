#include "neuralnetwork.h"
//===================================
//======= NeuralNetwork =================
//===================================
NeuralNetwork::NeuralNetwork(const std::vector<uint> &neurons)
    : num_of_neurons(std::move(neurons))
{
    uint nr_prev_layer_neurons = 0;
    for ( auto const &num : num_of_neurons )
    {
        layers.push_back(Layer(num, nr_prev_layer_neurons));
        nr_prev_layer_neurons = num;
    }


}

void NeuralNetwork::feedforward(const mat& input)
{
    layers[1].feedforward(input);
    for (uint var = 2; var < layers.size(); ++var)
    {
        layers[var].feedforward(layers[var - 1].neurons);
    }

    layers[layers.size() - 1].neurons.print("outputs: ");
}

void NeuralNetwork::mini_batch(mat x, mat y)
{
    feedforward(x);
    y.print("y;");

    uint output_layer = layers.size() - 1;

    mat error = 0.5 * (layers[output_layer].neurons - y) % (layers[output_layer].neurons - y);
    double er = 0;
    for (uint var = 0; var < error.size(); ++var) {
        er += error(var,0);
    }
    cout << "error: " << er << endl;

    // output error
    layers[output_layer].delta = grad_cost(y) % activation_derivate(output_layer);

    // backprop
    for (int var = output_layer - 1; var > 0 ; --var)
    {
        mat tmp = layers[var + 1].W.t() * layers[var + 1].delta;
        layers[var].delta = tmp % activation_derivate(var);
    }

    update_weights(x);
}

void NeuralNetwork::update_weights(mat input)
{
    //std::cout << "update weights\n";
    uint output_layer = layers.size();
    double learning_rate = .01;

    //layers[1].W.print("w11");
    layers[1].W = layers[1].W - (learning_rate * (layers[1].delta * input.t()));
    layers[1].W.print("w12");

    layers[1].bias.print("bias");
    layers[1].bias = layers[1].bias -  (learning_rate * layers[1].delta);

    for (uint var = 2; var < output_layer ; ++var)
    {
      /* network[var].W.print("W1");

        mat tmp = network[var].delta * network[var-1].neurons.t();

        network[var].delta.print("delta");
        network[var-1].neurons.print("a - 1");
        tmp.print("tmp");
*/

        layers[var].W = layers[var].W - (learning_rate * (layers[var].delta * layers[var-1].neurons.t()));
  //      network[var].W.print("W2");
        layers[var].bias = layers[var].bias -  (learning_rate * layers[var].delta);
    }
}

mat NeuralNetwork::grad_cost(mat y)
{
    //network[network.size() - 1].neurons.print("asd");
    return layers[layers.size() - 1].neurons - y;
}

mat NeuralNetwork::activation_derivate(int i_layer)
{
    // Output layer
    auto& layer = layers[i_layer];
    mat ret = zeros(layer.neurons.n_rows, 1);

    //cout << layer.neurons.n_rows << " " << layer.neurons.size() << endl;
    for (uint var = 0; var < layer.neurons.n_rows; ++var)
    {
        //ret(var, 0) = Activations::ReLu_derivative(layer.neurons[var]);
        ret(var, 0) = Activations::sigmoid_derivative(layer.neurons[var]);
    }
    return ret;
}


//===================================
//=========== Layer =================
//===================================




Layer::Layer(uint nr_neurons, uint nr_prev_layer_neurons)
    : nr_prev_layer_neurons(nr_prev_layer_neurons),
      neurons(ones<mat>(nr_neurons, 1))
{
    //print_neurons();
    init_weights_and_bias();
}

void Layer::init_weights_and_bias()
{
    std::cout << "Initialising layer\n";
    arma_rng::set_seed_random();
    W.zeros(neurons.size(), nr_prev_layer_neurons);
    xavier_init(neurons.size(), W);
    W.print("w: ");
    if (nr_prev_layer_neurons > 0 )
    {
        bias.zeros(neurons.size(), 1);
        xavier_init(neurons.size(), bias);
        bias.print("bias: ");
    }
}

void Layer::xavier_init(double nr_neuron, mat& matrix)
{
    double mean = 0.0;
    double variance = 1.0 / nr_neuron;

    for (uint col = 0; col < matrix.n_cols; ++col) {
        for (uint row = 0; row < matrix.n_rows; ++row) {
            matrix(row, col) = normal_dist_number(mean, variance);
        }
    }
}

double Layer::normal_dist_number(double mean, double variance)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, variance);
    return d(gen);
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
 /*   double denom = 0;
    for (uint var = 0; var < neurons.n_rows; ++var)
    {
        denom += std::exp( neurons(var, 0) );
    }

    for (uint var = 0; var < neurons.n_rows; ++var)
    {
        neurons(var, 0) = std::exp( neurons(var, 0) ) / denom ;
        cout << neurons(var, 0) << "asd \n";
    }
*/

    for (uint var = 0; var < neurons.n_rows; ++var)
    {
        //neurons(var, 0) = Activations::ReLu(neurons(var, 0));
        neurons(var, 0) = Activations::sigmoid(neurons(var, 0));
    }
}

void Layer::feedforward(const mat &input)
{
    neurons = W * input + bias;
   // neurons.print("nnnn");
    set_activation();
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

//====================================================
//====================Activations=====================
//====================================================

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
