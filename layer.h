#pragma once
#include <vector>

class layer {

public:

    bool dl_flag;
    bool cl_flag;
    bool pl_flag;

    bool sigmoid_flag;
    bool relu_flag;
    bool softmax_flag;

    std::vector<std::vector<double>> dl_forward_output;
    std::vector<std::vector<double>> dl_backward_output;

    std::vector<std::vector<std::vector<std::vector<double>>>> cl_forward_output;
    std::vector<std::vector<std::vector<std::vector<double>>>> cl_backward_output;

    void virtual forward(std::vector<std::vector<double>>& batched_input) {}
    void virtual forward(layer* prev_layer) = 0;
    void virtual forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_input) {}

    void virtual sigmoid_activation_function() {}
    void virtual rectified_linear_activation_function() {}
    void virtual softmax_activation_function() {}

    double virtual loss(std::vector<std::vector<double>>& batched_targets) { return 0.0; }
    double virtual loss(std::vector<int>& batched_targets) { return 0.0; }
    double virtual loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) { return 0.0; }

    void virtual init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {}
    void virtual init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets) {}
    void virtual init_backpropigation(layer* prev_layer, std::vector<std::vector<double>>& batched_targets) {}
    void virtual init_backpropigation(layer* prev_layer, std::vector<int>& batched_targets) {}
    void virtual init_backpropigation(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {}
    void virtual init_backpropigation(layer* prev_layer, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {}

    void virtual backward(std::vector<std::vector<double>>& batched_inputs) {}
    void virtual backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {}
    void virtual backward(layer* prev_layer) = 0;

    void virtual update_parameters(double learning_rate) {}

    std::vector<int> virtual get_shape() = 0;
    std::vector<std::vector<std::vector<std::vector<double>>>> virtual get_cl_weights() { return {}; }
    std::vector<std::vector<double>> virtual get_dl_weights() { return {}; }
    std::vector<double> virtual get_bais() { return {}; }
};


class dense_layer : public layer {

public:
    int neurons;
    int inputs;
    double sdg_mass;

    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> d_weights;
    std::vector<std::vector<double>> sgd_mass_weights;
    std::vector<double> bais;
    std::vector<double> d_bais;
    std::vector<double> sgd_mass_bais;

    dense_layer(int _inputs, int _neurons, double _sgd_mass);

    void virtual forward(std::vector<std::vector<double>>& batched_input) override;
    void virtual forward(layer* prev_layer) override;
    void virtual sigmoid_activation_function() override;
    void virtual rectified_linear_activation_function() override;
    void virtual softmax_activation_function() override;
    double virtual loss(std::vector<std::vector<double>>& batched_targets) override;
    double virtual loss(std::vector<int>& batched_targets) override;
    void virtual init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) override;
    void virtual init_backpropigation(layer* prev_layer, std::vector<std::vector<double>>& batched_targets) override;
    void virtual init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets) override;
    void virtual init_backpropigation(layer* prev_layer, std::vector<int>& batched_targets) override;
    void virtual backward(std::vector<std::vector<double>>& batched_inputs) override;
    void virtual backward(layer* prev_layer) override;
    void virtual update_parameters(double learning_rate) override;

    std::vector<int> virtual get_shape() override;
    std::vector<std::vector<double>> virtual get_dl_weights() override;
    std::vector<double> virtual get_bais() override;
};

class convolutional_layer : public layer {

public:
    int input_size;
    int input_depth;
    int kernals;
    int kernal_size;
    int padding;
    int stride;
    int output_size;

    double sdg_mass;

    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<std::vector<std::vector<std::vector<double>>>> d_weights;
    std::vector<std::vector<std::vector<std::vector<double>>>> sgd_mass_weights;

    std::vector<double> bais;
    std::vector<double> d_bais;
    std::vector<double> sgd_mass_bais;

    convolutional_layer(int _input_size, int _input_depth,int _kernals, int _kernal_size, int _padding, int _stride, double _sdg_mass);

    void virtual forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
    void virtual forward(layer* prev_layer) override;

    void virtual sigmoid_activation_function() override;
    void virtual rectified_linear_activation_function() override;

    double virtual loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;

    void virtual init_backpropigation(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
    void virtual init_backpropigation(layer* prev_layer, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
    void virtual backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
    void virtual backward(layer* prev_layer) override;
    void virtual update_parameters(double learning_rate) override;

    std::vector<int> virtual get_shape() override;
    std::vector<std::vector<std::vector<std::vector<double>>>> virtual get_cl_weights() override;
    std::vector<double> virtual get_bais() override;
};

class max_pooling_layer : public layer {

public:
    int input_size;
    int input_depth;
    int kernal_size;
    int output_size;
    int stride;

    max_pooling_layer(int _input_size, int _input_depth, int _kernal_size, int _stride);
    void virtual forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
    void virtual forward(layer* prev_layer) override;
    double virtual loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
    void virtual backward(layer* prev_layer) override;
    void virtual init_backpropigation(layer* prev_layer, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);

    std::vector<int> virtual get_shape() override;
};