#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "layer.h"

enum model_type { classification, regression, general_adversarial, res_net };

enum activation_function { linear, sigmoid, rectified_linear, softmax};

class model {

public:

	double learning_rate;

	model();
	model(int _model_type);
	model(int _model_type, double _learning_rate);
	model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass);

	void add_dense_layer(int _inputs, int _neurons, int _activation_function);
	void add_convolutional_layer(int _input_size, int _input_channels, int _kernals, int _kernal_size, int _padding, int _stride, int _activation_function);
	void add_pooling_layer(int _input_size, int _input_channels, int _kernal_size, int _padding, int _stride);
	
	void forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, int starting_layer, int ending_layer);
	void forward(std::vector<std::vector<double>>& batched_inputs, int starting_layer, int ending_layer);
	void forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs);
	void forward(std::vector<std::vector<double>>& batched_inputs);

	std::vector<std::vector<double>> dense_layer_output(int layer_index);
	std::vector<std::vector<std::vector<std::vector<double>>>> convolutional_layer_output(int layer_index);

	double loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
	double loss(std::vector<std::vector<double>>& batched_targets);
	double loss(std::vector<int>& batched_targets);

	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets, int starting_layer, int ending_layer);
	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets, int starting_layer, int ending_layer);
	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets, int starting_layer, int ending_layer);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets, int starting_layer, int ending_layer);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets, int starting_layer, int ending_layer);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets, int starting_layer, int ending_layer);

	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets);
	void backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets);
	void backward(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets);

	void update_parameters();
	void update_parameters(int starting_layer, int ending_layer);

	void train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
	void train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets);
	void train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets);
	void train(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets);
	void train(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets);
	void train(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets);

	void decay_learning_rate();

	void save_model(const std::string& file_name);
	void load_model(const std::string& file_name);

	std::vector<int> debug(int i) { return layers[i]->get_shape(); }

	~model();
private:
	int type;
	int layer_count;
	int step;
	double starting_learning_rate;
	double decay_rate;
	double sgd_mass;
	
	std::vector<int> activation_functions;
	std::vector<layer*> layers;
	
	void init_model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass);

	void write_dense_layer(std::ofstream& file, int layer_idx);
	void write_convolutional_layer(std::ofstream& file, int layer_idx);
	void write_pooling_layer(std::ofstream& file, int layer_idx);
};
