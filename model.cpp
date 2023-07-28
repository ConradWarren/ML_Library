#include "model.h"
#include <string>
#include <iostream>
#include <fstream>

void model::init(int _type, double _learning_rate, double _decay_rate, double _sgd_mass) {
	gan_layer_boundry = -1;
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

void model::forward(std::vector<std::vector<double>>& batched_data, int start_point, int end_point) {

	if (layers.size() <= start_point) {
		return;
	}

	layers[start_point]->forward(batched_data);
	if (activation_functions[start_point] == activation_function::rectified_linear) layers[start_point]->rectified_linear_activation_function();

	else if (activation_functions[start_point] == activation_function::sigmoid) layers[start_point]->sigmoid_activation_function();

	for (int i = start_point+1; i <= end_point; i++) {

		layers[i]->forward(layers[i - 1]->output);

		if (activation_functions[i] == activation_function::rectified_linear) layers[i]->rectified_linear_activation_function();

		else if (activation_functions[i] == activation_function::sigmoid) layers[i]->sigmoid_activation_function();

		else if (activation_functions[i] == activation_function::softmax) {
			//softmax goes here
		}
	}
}

void model::forward(std::vector<std::vector<double>>& batched_data) {
	forward(batched_data, 0, layer_count-1);
}

void model::backward(std::vector<std::vector<double>>& batched_targets,std::vector<std::vector<double>>& batched_data, int start_point, int end_point) {

	if (start_point >= layer_count) {
		return;
	}

	if (start_point > 0) {
		layers[start_point]->init_back_propagation(layers[start_point - 1]->output, batched_targets);
	}
	else {
		layers[start_point]->init_back_propagation(batched_data, batched_targets);
	}

	for (int i = start_point - 1; i >= end_point+1; i--) {
		layers[i]->backward(layers[i - 1]->output, layers[i + 1]->weights, layers[i + 1]->output);
	}

	if (start_point > 0) {
		layers[end_point]->backward(batched_data, layers[end_point+1]->weights, layers[end_point+1]->output);
	}
}


void model::backward(std::vector<std::vector<double>>& batched_targets, std::vector<std::vector<double>>& batched_data) {
	backward(batched_targets, batched_data, layer_count - 1, 0);
}

void model::train(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss) {
	
	if (layers.empty()) {
		return;
	}

	forward(batched_data);

	if (print_loss) {
		std::cout << loss(batched_targets) << '\n';
	}

	backward(batched_targets, batched_data);

	update_parameters();
}

void model::train_gan_classifier(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& real_data, std::vector<std::vector<double>>& batch_lables, bool print_loss) {

	if (layers.size() < 2) {
		return;
	}

	if (type != model_type::general_adversarial) {
		std::cout << "not valid model type\n";
		return;
	}

	if (gan_layer_boundry == -1) {
		std::cout << "layer boundry not set\n";
		return;
	}
	
	forward(batched_data, 0, gan_layer_boundry);
	int idx = 0;
	std::vector<std::vector<double>> classifier_training_batch(layers[gan_layer_boundry-1]->output.size() + real_data.size());

	for (int i = 0; i < layers[gan_layer_boundry - 1]->output.size(); i++) {
		classifier_training_batch[idx] = layers[gan_layer_boundry - 1]->output[i];
		idx++;
	}
	for (int i = 0; i < real_data.size(); i++) {
		classifier_training_batch[idx] = real_data[i];
		idx++;
	}

	forward(classifier_training_batch, gan_layer_boundry, layer_count);
	
	if (print_loss) {
		std::cout << layers[layer_count - 1]->loss(batch_lables) << '\n';
	}

	backward(batch_lables, classifier_training_batch, layer_count - 1, gan_layer_boundry);
	
	update_parameters(gan_layer_boundry, layer_count - 1);
}

void model::train_gan_classifier(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& real_data, bool print_loss) {

	std::vector<std::vector<double>> batch_lables(batched_data.size() + real_data.size(), std::vector<double>(1, 0.0));

	for (int i = batched_data.size(); i < batch_lables.size(); i++) {
		batch_lables[i][0] = 1.0;
	}

	train_gan_classifier(batched_data, real_data, batch_lables, print_loss);
}

void model::train_gan_generator(std::vector<std::vector<double>>& batched_data, std::vector<std::vector<double>>& batched_targets, bool print_loss) {

	if (layers.size() < 2) {
		return;
	}

	if (type != model_type::general_adversarial) {
		std::cout << "not valid model type\n";
		return;
	}

	if (gan_layer_boundry == -1) {
		std::cout << "layer boundry not set\n";
		return;
	}

	forward(batched_data);

	if (print_loss) {
		std::cout << loss(batched_targets) << '\n';
	}

	backward(batched_targets, batched_data);
	
	update_parameters(0, gan_layer_boundry - 1);
}

void model::train_gan_generator(std::vector<std::vector<double>>& batched_data, bool print_loss) {

	std::vector<std::vector<double>> batched_targets(batched_data.size(), std::vector<double>(1, 1.0));

	train_gan_generator(batched_data, batched_targets, print_loss);
}

void model::update_parameters(int start_point, int end_point) {
	for (int i = start_point; i <= end_point; i++) {
		layers[i]->train(learning_rate);
	}
}

void model::update_parameters() {
	update_parameters(0, layer_count - 1);
}

void model::decay() {
	step++;
	learning_rate = starting_learning_rate / (1.0 + (decay_rate * step));
}

double model::loss(std::vector<std::vector<double>>& batched_targets) {
	
	return layers.back()->loss(batched_targets);
}

void model::set_layer_boundry(int _gan_layer_boundry) {
	
	if (type != model_type::general_adversarial) {
		std::cout << "not valid model type\n";
		return;
	}
	gan_layer_boundry = _gan_layer_boundry;
}

void model::set_layer_boundry() {
	set_layer_boundry(layer_count);
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

void model::load_output(int layer_idx) {
	output = layers[layer_idx]->output;
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
