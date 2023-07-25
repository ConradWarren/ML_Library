#include "model.h"
#include <string>
#include <iostream>
#include <fstream>

void model::init(int _type, double _learning_rate, double _decay_rate, double _sgd_mass) {
	type = _type;
	starting_learning_rate = _learning_rate;
	learning_rate = _learning_rate;
	decay_rate = _decay_rate;
	sgd_mass = _sgd_mass;
	step = 0;
	layer_count = 0;
}

model::model() {
	init(model_type::regression, 1e-3, 0.0, 0.0);
}
model::model(int _model_type) {
	init(_model_type, 1e-3, 0.0, 0.0);
}
model::model(int _model_type, double _learning_rate) {
	init(_model_type, _learning_rate, 0.0, 0.0);
}
model::model(int _model_type, double _learning_rate, double _decay_rate) {
	init(_model_type, _learning_rate, _decay_rate, 0.0);
}
model::model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass) {
	init(_model_type, _learning_rate, _decay_rate, _sgd_mass);
}

void model::add_layer(int inputs, int neurons, int _activation_function) {
	layers.push_back(new layer(inputs, neurons, sgd_mass));
	activation_functions.push_back(_activation_function);
	layer_count++;
}

void model::add_layer(int inputs, int neurons) {
	add_layer(inputs, neurons, activation_function::linear);
}

void model::forward(std::vector<std::vector<double>>& batched_data) {

	if (layers.empty()) {
		return;
	}

	layers[0]->forward(batched_data);
	if (activation_functions[0] == activation_function::rectified_linear) layers[0]->rectified_linear_activation_function();

	else if (activation_functions[0] == activation_function::sigmoid) layers[0]->sigmoid_activation_function();

	for (int i = 1; i < layer_count; i++) {

		layers[i]->forward(layers[i - 1]->output);

		if (activation_functions[i] == activation_function::rectified_linear) layers[i]->rectified_linear_activation_function();

		else if (activation_functions[i] == activation_function::sigmoid) layers[i]->sigmoid_activation_function();

		else if (activation_functions[i] == activation_function::softmax) {
			//softmax goes here
		}
	}
}

void model::train(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss) {
	
	if (layers.empty()) {
		return;
	}

	forward(batched_data);

	if (print_loss) {
		std::cout << loss(batched_targets) << '\n';
	}

	if (layer_count > 1) {
		layers[layer_count - 1]->init_back_propagation(layers[layer_count - 2]->output, batched_targets);
	}
	else {
		layers[layer_count - 1]->init_back_propagation(batched_data, batched_targets);
	}
	for (int i = layer_count - 2; i >= 1; i--) {
		layers[i]->backward(layers[i - 1]->output, layers[i + 1]->weights, layers[i + 1]->output);
	}
	if (layer_count > 1) {
		layers[0]->backward(batched_data, layers[1]->weights, layers[1]->output);
	}

	update_parameters();
}

void model::update_parameters() {
	for (int i = 0; i < layer_count; i++) {
		layers[i]->train(learning_rate);
	}
}

void model::decay() {
	step++;
	learning_rate = starting_learning_rate / (1.0 + (decay_rate * step));
}

double model::loss(std::vector<std::vector<double>>& batched_targets) {
	return layers.back()->loss(batched_targets);
}

void model::print_output() {
	
	for (int i = 0; i < layers.back()->output.size(); i++) {
		std::cout << "{";
		for (int j = 0; j < layers.back()->output[i].size(); j++) {
			std::cout << layers.back()->output[i][j];
			if (j != layers.back()->output[i].size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "}\n";
	}
}

void model::save_weights(std::string file_name) {

	std::ofstream file(file_name, std::ios::out);

	if (!file.is_open()) {
		std::cout << "could not open file\n";
		return;
	}

	for (int i = 0; i < layers.size(); i++) {

		file << "layer_shape," << layers[i]->inputs << ',' << layers[i]->neurons << ','<<activation_functions[i]<<",\n";

		for (int j = 0; j < layers[i]->neurons; j++) {

			for (int x = 0; x < layers[i]->inputs; x++) {
				file << layers[i]->weights[j][x] << ',';
			}
			file << '\n';
		}

		for (int j = 0; j < layers[i]->neurons; j++) {
			file << layers[i]->bais[j] << ',';
		}
		file << '\n';
	}

	file.close();
}

void model::load_output() {
	output = layers.back()->output;
}

void model::load_weights(std::string file_name) {

	std::ifstream file(file_name);
	std::string current_line;
	std::string current_parameter;
	std::vector<int> shape;

	if (!file.is_open()) {
		std::cout << "could not open file\n";
		return;
	}

	std::getline(file, current_line);
	while (!current_line.empty()) {

		if (current_line.substr(0, 12) != "layer_shape,") {
			std::cout << "file not formated correctly\n";
			return;
		}
		shape.clear();
		for (int i = 12; i < current_line.length(); i++) {

			if (current_line[i] == ',' && !current_parameter.empty()) {
				shape.push_back(std::stoi(current_parameter));
				current_parameter.clear();
			}
			else if(current_line[i] != ',') {
				current_parameter += current_line[i];
			}
		}
		
		add_layer(shape[0], shape[1], shape[2]);

		for (int i = 0; i < shape[1]; i++) {

			std::getline(file, current_line);
			
			int idx = 0;
			for (int j = 0; j < current_line.length(); j++){
				if (current_line[j] == ',' && !current_parameter.empty()) {
					layers.back()->weights[i][idx] = std::stold(current_parameter);
					current_parameter.clear();
					idx++;
				}
				else if (current_line[j] != ',') {
					current_parameter += current_line[j];
				}
			}
		}

		int idx = 0;
		std::getline(file, current_line);

		for (int i = 0; i < current_line.length(); i++) {
			if (current_line[i] == ',' && !current_parameter.empty()) {
				layers.back()->bais[idx] = std::stod(current_parameter);
				current_parameter.clear();
				idx++;
			}
			else if (current_line[i] != ',') {
				current_parameter += current_line[i];
			}
		}
		std::getline(file, current_line);
	}


	file.close();
}

model::~model() {
	for (int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}

