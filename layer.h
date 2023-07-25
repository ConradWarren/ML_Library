#pragma once
#include <vector>

class layer {

public:
	int neurons;
	int inputs;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> d_weights;
	std::vector<std::vector<double>> weight_momentums;
	std::vector<double> bais;
	std::vector<double> d_bais;
	std::vector<double> bais_momentums;
	std::vector<std::vector<double>> output;
	double sgd_mass;
	bool sigmoid_flag;
	bool relu_flag;

	layer(int _inputs, int _neurons, double _sgd_mass);
	
	void forward(std::vector<std::vector<double>>& batched_inputs);
	
	void sigmoid_activation_function();

	void rectified_linear_activation_function();

	double loss(std::vector<std::vector<double>>& batched_targets);

	void init_back_propagation(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets);

	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& forward_weights, std::vector<std::vector<double>>& forward_output);
	
	void train(double learning_rate);
};
