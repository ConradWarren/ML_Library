#include "layer.h"
#include <random>
#include <iostream>

static const double e = 2.71828182846;

dense_layer::dense_layer(int _inputs, int _neurons, double _sgd_mass) {

	cl_flag = false;
	dl_flag = true;
	pl_flag = false;

	inputs = _inputs;
	neurons = _neurons;
	sdg_mass = _sgd_mass;

	sigmoid_flag = false;
	relu_flag = false;
	softmax_flag = false;

	weights.resize(neurons, std::vector<double>(inputs));
	d_weights.resize(neurons, std::vector<double>(inputs, 0));
	sgd_mass_weights.resize(neurons, std::vector<double>(inputs, 0));

	bais.resize(neurons);
	d_bais.resize(neurons,0);
	sgd_mass_bais.resize(neurons,0);

	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (int i = 0; i < neurons; i++) {
		for (int j = 0; j < inputs; j++) {
			weights[i][j] = distribution(generator);
		}
		bais[i] = distribution(generator);
	}
}

void dense_layer::forward(std::vector<std::vector<double>>& batched_input) {

	if (dl_forward_output.size() != batched_input.size()) {
		dl_forward_output.resize(batched_input.size(), std::vector<double>(neurons));
	}

	for (int batch_num = 0; batch_num < batched_input.size(); batch_num++) {
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_forward_output[batch_num][neuron_num] = 0;
			for (int input_num = 0; input_num < inputs; input_num++) {
				dl_forward_output[batch_num][neuron_num] += weights[neuron_num][input_num] * batched_input[batch_num][input_num];
			}
			dl_forward_output[batch_num][neuron_num] += bais[neuron_num];
		}
	}
}


void dense_layer::forward(layer* prev_layer){

	if (prev_layer->dl_flag) {
		forward(prev_layer->dl_forward_output);
		return;
	}
	
	if (prev_layer->dl_forward_output.size() != prev_layer->cl_forward_output.size()) {
		prev_layer->dl_forward_output.resize(prev_layer->cl_forward_output.size(), std::vector<double>(inputs));
	}

	int idx = 0;
	for (int batch_num = 0; batch_num < prev_layer->cl_forward_output.size(); batch_num++) {
		idx = 0;
		for (int filter_num = 0; filter_num < prev_layer->cl_forward_output[0].size(); filter_num++) {
			for (int y = 0; y < prev_layer->cl_forward_output[0][0].size(); y++) {
				for (int x = 0; x < prev_layer->cl_forward_output[0][0][0].size(); x++) {
					prev_layer->dl_forward_output[batch_num][idx] = prev_layer->cl_forward_output[batch_num][filter_num][y][x];
					idx++;
					
				}
			}
		}
	}

	forward(prev_layer->dl_forward_output);
}

void dense_layer::sigmoid_activation_function() {

	sigmoid_flag = true;

	for (int batch_num = 0; batch_num < dl_forward_output.size(); batch_num++) {
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_forward_output[batch_num][neuron_num] = 1.0 / (1.0 + std::pow(e, -dl_forward_output[batch_num][neuron_num]));
		}
	}
}

void dense_layer::rectified_linear_activation_function() {

	relu_flag = true;

	for (int batch_num = 0; batch_num < dl_forward_output.size(); batch_num++) {
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_forward_output[batch_num][neuron_num] = (dl_forward_output[batch_num][neuron_num] > 0.0) ? dl_forward_output[batch_num][neuron_num] : 0.0;
		}
	}
}

void dense_layer::softmax_activation_function() {

	softmax_flag = true;
	double sum = 0;

	for (int batch_num = 0; batch_num < dl_forward_output.size(); batch_num++) {
		sum = 0;

		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_forward_output[batch_num][neuron_num] = std::pow(e, dl_forward_output[batch_num][neuron_num]);
			sum += dl_forward_output[batch_num][neuron_num];
		}

		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_forward_output[batch_num][neuron_num] /= sum;
		}
	}
}

double dense_layer::loss(std::vector<std::vector<double>>& batched_targets) {

	double result = 0.0;
	double batch_size = double(batched_targets.size());

	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			result += ((dl_forward_output[batch_num][neuron_num] - batched_targets[batch_num][neuron_num]) * (dl_forward_output[batch_num][neuron_num] - batched_targets[batch_num][neuron_num])) / batch_size;
		}
	}
	return result;
}

double dense_layer::loss(std::vector<int>& batched_targets) {

	double result = 0.0;
	double batch_size = double(batched_targets.size());

	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {

		if (dl_forward_output[batch_num][batched_targets[batch_num]] == 0.0) {
			result -= ((std::log(dl_forward_output[batch_num][batched_targets[batch_num]] + 1e-60)) / batch_size);
		}
		else if (dl_forward_output[batch_num][batched_targets[batch_num]] == 1.0) {
			result -= ((std::log(dl_forward_output[batch_num][batched_targets[batch_num]] - 1e-60)) / batch_size);
		}
		else {
			result -= ((std::log(dl_forward_output[batch_num][batched_targets[batch_num]])) / batch_size);
		}

		
	}

	return result;
}

void dense_layer::init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {

	if (dl_backward_output.size() != dl_forward_output.size()) {
		dl_backward_output.resize(dl_forward_output.size(), std::vector<double>(neurons));
	}

	for (int batch_num = 0; batch_num < dl_backward_output.size(); batch_num++) {
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {

			dl_backward_output[batch_num][neuron_num] = (dl_forward_output[batch_num][neuron_num] - batched_targets[batch_num][neuron_num]) * 2.0 * (1.0/double(batched_inputs.size()));
			
			if (sigmoid_flag) {
				dl_backward_output[batch_num][neuron_num] *= (dl_forward_output[batch_num][neuron_num] * (1 - dl_forward_output[batch_num][neuron_num]));
			}
			else if (relu_flag && dl_forward_output[batch_num][neuron_num] == 0.0) {
				dl_backward_output[batch_num][neuron_num] = 0.0;
			}
		}
	}

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		for (int input_num = 0; input_num < inputs; input_num++) {
			d_weights[neuron_num][input_num] = 0.0;
			for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
				d_weights[neuron_num][input_num] += batched_inputs[batch_num][input_num] * dl_backward_output[batch_num][neuron_num];
			}
		}
	}

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		d_bais[neuron_num] = 0.0;
		for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
			d_bais[neuron_num] += dl_backward_output[batch_num][neuron_num];
		}
	}
}

void dense_layer::init_backpropigation(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets) {

	if (dl_backward_output.size() != dl_forward_output.size()) {
		dl_backward_output.resize(dl_forward_output.size(), std::vector<double>(neurons));
	}

	for(int batch_num = 0; batch_num < dl_backward_output.size(); batch_num++){
		for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
			dl_backward_output[batch_num][neuron_num] = dl_forward_output[batch_num][neuron_num];
			if (batched_targets[batch_num] == neuron_num) {
				dl_backward_output[batch_num][neuron_num] -= 1.0;
			}
			dl_backward_output[batch_num][neuron_num] /= double(dl_backward_output.size());
		}
	}

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		for (int input_num = 0; input_num < inputs; input_num++) {
			d_weights[neuron_num][input_num] = 0.0;
			for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
				d_weights[neuron_num][input_num] += batched_inputs[batch_num][input_num] * dl_backward_output[batch_num][neuron_num];
			}
		}
	}

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		d_bais[neuron_num] = 0.0;
		for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
			d_bais[neuron_num] += dl_backward_output[batch_num][neuron_num];
		}
	}
}

void dense_layer::init_backpropigation(layer* prev_layer, std::vector<int>& batched_targets) {

	if (prev_layer->cl_flag) {
		if (prev_layer->dl_forward_output.size() != prev_layer->cl_forward_output.size()) {
			dl_forward_output.resize(cl_forward_output.size(), std::vector<double>(inputs));
		}
		int idx = 0;
		for (int batch_num = 0; batch_num < prev_layer->cl_forward_output.size(); batch_num++) {
			idx = 0;
			for (int filter_num = 0; filter_num < prev_layer->cl_forward_output[0].size(); filter_num++) {
				for (int y = 0; y < prev_layer->cl_forward_output[0][0].size(); y++) {
					for (int x = 0; x < prev_layer->cl_forward_output[0][0][0].size(); x++) {
						prev_layer->dl_forward_output[batch_num][idx] = prev_layer->cl_forward_output[batch_num][filter_num][y][x];
						idx++;
					}
				}
			}
		}
	}
	init_backpropigation(prev_layer->dl_forward_output, batched_targets);

	if (prev_layer->dl_backward_output.size() != batched_targets.size()) {
		prev_layer->dl_backward_output.resize(batched_targets.size(), std::vector<double>(inputs));
	}

	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {

		for (int input_num = 0; input_num < inputs; input_num++) {

			prev_layer->dl_backward_output[batch_num][input_num] = 0.0;

			for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
				prev_layer->dl_backward_output[batch_num][input_num] += dl_backward_output[batch_num][neuron_num] * weights[neuron_num][input_num];
			}

			if (prev_layer->sigmoid_flag) {
				prev_layer->dl_backward_output[batch_num][input_num] *= (prev_layer->dl_forward_output[batch_num][input_num] * (1 - prev_layer->dl_forward_output[batch_num][input_num]));
			}
			else if (prev_layer->relu_flag && prev_layer->dl_forward_output[batch_num][input_num] == 0.0) {
				prev_layer->dl_backward_output[batch_num][input_num] = 0.0;
			}

		}
	}

	if (prev_layer->dl_flag) {
		return;
	}

	if (prev_layer->cl_backward_output.size() != prev_layer->dl_backward_output.size()) {
		prev_layer->cl_backward_output.resize(prev_layer->cl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(prev_layer->cl_forward_output[0].size(), std::vector<std::vector<double>>(prev_layer->cl_forward_output[0][0].size(), std::vector<double>(prev_layer->cl_forward_output[0][0].size()))));
	}
	int idx = 0;
	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
		idx = 0;
		for (int channel_num = 0; channel_num < prev_layer->cl_backward_output[0].size(); channel_num++) {
			for (int y = 0; y < prev_layer->cl_backward_output[0][0].size(); y++) {
				for (int x = 0; x < prev_layer->cl_backward_output[0][0][0].size(); x++) {
					prev_layer->cl_backward_output[batch_num][channel_num][y][x] = prev_layer->dl_backward_output[batch_num][idx];
					idx++;

				}
			}
		}
	}
}

void dense_layer::init_backpropigation(layer* prev_layer, std::vector<std::vector<double>>& batched_targets) {

	if(prev_layer->cl_flag) {
		if (prev_layer->dl_forward_output.size() != prev_layer->cl_forward_output.size()) {
			dl_forward_output.resize(cl_forward_output.size(), std::vector<double>(inputs));
		}
		int idx = 0;
		for (int batch_num = 0; batch_num < prev_layer->cl_forward_output.size(); batch_num++) {
			idx = 0;
			for (int filter_num = 0; filter_num < prev_layer->cl_forward_output[0].size(); filter_num++) {
				for (int y = 0; y < prev_layer->cl_forward_output[0][0].size(); y++) {
					for (int x = 0; x < prev_layer->cl_forward_output[0][0][0].size(); x++) {
						prev_layer->dl_forward_output[batch_num][idx] = prev_layer->cl_forward_output[batch_num][filter_num][y][x];
						idx++;
					}
				}
			}
		}
	}
	init_backpropigation(prev_layer->dl_forward_output, batched_targets);

	if (prev_layer->dl_backward_output.size() != batched_targets.size()) {
		prev_layer->dl_backward_output.resize(batched_targets.size(), std::vector<double>(inputs));
	}

	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {

		for (int input_num = 0; input_num < inputs; input_num++) {

			prev_layer->dl_backward_output[batch_num][input_num] = 0.0;

			for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
				prev_layer->dl_backward_output[batch_num][input_num] += dl_backward_output[batch_num][neuron_num] * weights[neuron_num][input_num];
			}

			if (prev_layer->sigmoid_flag) {
				prev_layer->dl_backward_output[batch_num][input_num] *= (prev_layer->dl_forward_output[batch_num][input_num] * (1 - prev_layer->dl_forward_output[batch_num][input_num]));
			}
			else if (prev_layer->relu_flag && prev_layer->dl_forward_output[batch_num][input_num] == 0.0) {
				prev_layer->dl_backward_output[batch_num][input_num] = 0.0;
			}

		}
	}

	if (prev_layer->dl_flag) {
		return;
	}
	
	if (prev_layer->cl_backward_output.size() != prev_layer->dl_backward_output.size()) {
		prev_layer->cl_backward_output.resize(prev_layer->cl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(prev_layer->cl_forward_output[0].size(), std::vector<std::vector<double>>(prev_layer->cl_forward_output[0][0].size(), std::vector<double>(prev_layer->cl_forward_output[0][0].size()))));
	}
	int idx = 0;
	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
		idx = 0;
		for (int channel_num = 0; channel_num < prev_layer->cl_backward_output[0].size(); channel_num++) {
			for (int y = 0; y < prev_layer->cl_backward_output[0][0].size(); y++) {
				for (int x = 0; x < prev_layer->cl_backward_output[0][0][0].size(); x++) {
					prev_layer->cl_backward_output[batch_num][channel_num][y][x] = prev_layer->dl_backward_output[batch_num][idx];
					idx++;
					
				}
			}
		}
	}
}

void dense_layer::backward(std::vector<std::vector<double>>& batched_inputs) {

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		for (int input_num = 0; input_num < inputs; input_num++) {
			d_weights[neuron_num][input_num] = 0.0;
			for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
				d_weights[neuron_num][input_num] += batched_inputs[batch_num][input_num] * dl_backward_output[batch_num][neuron_num];
			}
		}
	}

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		d_bais[neuron_num] = 0.0;
		for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
			d_bais[neuron_num] += dl_backward_output[batch_num][neuron_num];
		}
	}
}

void dense_layer::backward(layer* prev_layer) {

	if (prev_layer->cl_flag) {
		if (prev_layer->dl_forward_output.size() != prev_layer->cl_forward_output.size()) {
			dl_forward_output.resize(cl_forward_output.size(), std::vector<double>(inputs));
		}
		int idx = 0;
		for (int batch_num = 0; batch_num < prev_layer->cl_forward_output.size(); batch_num++) {
			idx = 0;
			for (int filter_num = 0; filter_num < prev_layer->cl_forward_output[0].size(); filter_num++) {
				for (int y = 0; y < prev_layer->cl_forward_output[0][0].size(); y++) {
					for (int x = 0; x < prev_layer->cl_forward_output[0][0][0].size(); x++) {
						prev_layer->dl_forward_output[batch_num][idx] = prev_layer->cl_forward_output[batch_num][filter_num][y][x];
						idx++;
					}
				}
			}
		}
	}
	backward(prev_layer->dl_forward_output);

	if (prev_layer->dl_backward_output.size() != dl_backward_output.size()) {
		prev_layer->dl_backward_output.resize(dl_backward_output.size(), std::vector<double>(inputs));
	}

	for (int batch_num = 0; batch_num < dl_backward_output.size(); batch_num++) {

		for (int input_num = 0; input_num < inputs; input_num++) {

			prev_layer->dl_backward_output[batch_num][input_num] = 0.0;

			for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
				prev_layer->dl_backward_output[batch_num][input_num] += dl_backward_output[batch_num][neuron_num] * weights[neuron_num][input_num];
			}

			if (prev_layer->sigmoid_flag) {
				prev_layer->dl_backward_output[batch_num][input_num] *= (prev_layer->dl_forward_output[batch_num][input_num] * (1 - prev_layer->dl_forward_output[batch_num][input_num]));
			}
			else if (prev_layer->relu_flag && prev_layer->dl_forward_output[batch_num][input_num] == 0.0) {
				prev_layer->dl_backward_output[batch_num][input_num] = 0.0;
			}

		}
	}
	if (prev_layer->dl_flag) {
		return;
	}

	if (prev_layer->cl_backward_output.size() != prev_layer->dl_backward_output.size()) {
		prev_layer->cl_backward_output.resize(prev_layer->cl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(prev_layer->cl_forward_output[0].size(), std::vector<std::vector<double>>(prev_layer->cl_forward_output[0][0].size(), std::vector<double>(prev_layer->cl_forward_output[0][0].size()))));
	}
	int idx = 0;
	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
		idx = 0;
		for (int channel_num = 0; channel_num < prev_layer->cl_backward_output[0].size(); channel_num++) {
			for (int y = 0; y < prev_layer->cl_backward_output[0][0].size(); y++) {
				for (int x = 0; x < prev_layer->cl_backward_output[0][0][0].size(); x++) {
					prev_layer->cl_backward_output[batch_num][channel_num][y][x] = prev_layer->dl_backward_output[batch_num][idx];
					idx++;

				}
			}
		}
	}
}

void dense_layer::update_parameters(double learning_rate) {

	double parameter_update = 0.0;

	for (int neuron_num = 0; neuron_num < neurons; neuron_num++) {
		for (int input_num = 0; input_num < inputs; input_num++) {
			parameter_update = (sdg_mass * sgd_mass_weights[neuron_num][input_num]) - (d_weights[neuron_num][input_num] * learning_rate);
			weights[neuron_num][input_num] += parameter_update;
			sgd_mass_weights[neuron_num][input_num] = parameter_update;
		}

		parameter_update = (sdg_mass * sgd_mass_bais[neuron_num]) - (d_bais[neuron_num] * learning_rate);
		bais[neuron_num] += parameter_update;
		sgd_mass_bais[neuron_num] = parameter_update;
	}
}

std::vector<int> dense_layer::get_shape() {
	return {inputs, neurons};
}

std::vector<std::vector<double>> dense_layer::get_dl_weights() {
	return weights;
}
std::vector<double> dense_layer::get_bais() {
	return bais;
}
