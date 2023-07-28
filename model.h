#pragma once
#include <string>
#include "layer.h"

enum model_type {classification, regression, general_adversarial};

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

	void forward(std::vector<std::vector<double>>& batched_data, int start_point, int end_point);
	void forward(std::vector<std::vector<double>>& batched_data);

	void backward(std::vector<std::vector<double>>& batched_targets, std::vector<std::vector<double>>& batched_data, int start_point, int end_point);
	void backward(std::vector<std::vector<double>>& batched_targets, std::vector<std::vector<double>>& batched_data);
	
	void train(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss);
	
	void train_gan_classifier(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& real_data, std::vector<std::vector<double>>& batch_lables, bool print_loss);
	void train_gan_classifier(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& real_data, bool print_loss);

	void train_gan_generator(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss);
	void train_gan_generator(std::vector<std::vector<double>>& batched_data, bool print_loss);

	void decay();

	double loss(std::vector<std::vector<double>>& batched_targets);

	void set_layer_boundry(int _gan_layer_boundry);

	void set_layer_boundry();

	void load_output();

	void load_output(int layer_idx);

	void update_parameters(int start_point, int end_point);
	void update_parameters();

	void print_output();

	void save_weights(std::string file_name);

	void load_weights(std::string file_name);

	~model();

private:

	int type;
	int layer_count;
	int gan_layer_boundry;
	std::vector<layer*> layers;
	std::vector<int> activation_functions;
	double starting_learning_rate;
	double learning_rate;
	double decay_rate;
	double sgd_mass;
	double step;

	void init(int _type, double _learning_rate, double _decay_rate, double sgd_mass);
};
