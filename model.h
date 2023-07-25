#pragma once
#include <string>
#include "layer.h"

enum model_type {classification, regression};

enum activation_function{linear, sigmoid, softmax, rectified_linear};

class model {

public:

	std::vector<std::vector<double>> output;

	model();
	model(int _model_type);
	model(int _model_type, double _learning_rate);
	model(int _model_type, double _learning_rate, double _decay_rate);
	model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass);

	void add_layer(int inputs, int neurons, int _activation_function);
	
	void add_layer(int inputs, int neurons);

	void train(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss);

	void decay();

	void forward(std::vector<std::vector<double>>& batched_data);

	double loss(std::vector<std::vector<double>>& batched_targets);

	void load_output();

	void print_output();

	void save_weights(std::string file_name);

	void load_weights(std::string file_name);

	~model();

private:

	int type;
	int layer_count;
	std::vector<layer*> layers;
	std::vector<int> activation_functions;
	double starting_learning_rate;
	double learning_rate;
	double decay_rate;
	double sgd_mass;
	double step;

	void init(int _type, double _learning_rate, double _decay_rate, double sgd_mass);

	void update_parameters();
};
