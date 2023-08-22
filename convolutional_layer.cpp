#include "layer.h"

#include <iostream>
#include <random>

static const double e = 2.71828182846;

convolutional_layer::convolutional_layer(int _input_size, int _input_depth, int _kernals, int _kernal_size, int _padding, int _stride, double _sdg_mass) {

	input_size = _input_size;
	input_depth = _input_depth;
	kernals = _kernals;
	kernal_size = _kernal_size;
	padding = _padding;
	stride = _stride;
	output_size = ((input_size + (2 * padding) - kernal_size) / stride) + 1;

	sdg_mass = _sdg_mass;

	sigmoid_flag = false;
	relu_flag = false;
	softmax_flag = false;
	dl_flag = false;
	cl_flag = true;
	pl_flag = false;

	weights.resize(kernals, std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(kernal_size, std::vector<double>(kernal_size))));
	d_weights.resize(kernals, std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(kernal_size, std::vector<double>(kernal_size))));
	sgd_mass_weights.resize(kernals, std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(kernal_size, std::vector<double>(kernal_size, 0))));

	bais.resize(kernals);
	d_bais.resize(kernals);
	sgd_mass_bais.resize(kernals, 0);

	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < kernal_size; y++) {
				for (int x = 0; x < kernal_size; x++) {
					weights[kernal_num][channel_num][y][x] = distribution(generator);
				}
			}
		}
		bais[kernal_num] = distribution(generator);
	}
}

void convolutional_layer::forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	if (cl_forward_output.size() != batched_inputs.size()) {
		cl_forward_output.resize(batched_inputs.size(), std::vector<std::vector<std::vector<double>>>(kernals, std::vector<std::vector<double>>(output_size, std::vector<double>(output_size, 0))));
	}
	else {
		for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
			for (int channel_num = 0; channel_num < kernals; channel_num++) {
				for (int y = 0; y < output_size; y++) {
					for (int x = 0; x < output_size; x++) {
						cl_forward_output[batch_num][channel_num][y][x] = 0.0;
					}
				}
			}
		}
	}

	int output_y_idx = 0;
	int output_x_idx = 0;

	for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {

		for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
			output_y_idx = 0;
			for (int y = 0 - padding; y < input_size - kernal_size + 1 + padding; y += stride) {

				output_x_idx = 0;
				for (int x = 0 - padding; x < input_size - kernal_size + 1 + padding; x += stride) {

					for (int a = 0; a < kernal_size; a++) {

						if (y + a < 0 || y + a >= input_size) {
							continue;
						}

						for (int b = 0; b < kernal_size; b++) {

							if (x + b < 0 || x + b >= input_size) {
								continue;
							}

							for (int channel_num = 0; channel_num < input_depth; channel_num++) {
								cl_forward_output[batch_num][kernal_num][output_y_idx][output_x_idx] += weights[kernal_num][channel_num][a][b] * batched_inputs[batch_num][channel_num][y + a][x + b];
							}

						}
					}
					cl_forward_output[batch_num][kernal_num][output_y_idx][output_x_idx] += bais[kernal_num];
					output_x_idx++;
				}
				output_y_idx++;
			}
		}
	}
}

void convolutional_layer::forward(layer* prev_layer) {
	
	if (prev_layer->dl_flag) {
		
		if (prev_layer->cl_forward_output.size() != prev_layer->dl_forward_output.size()) {
			prev_layer->cl_forward_output.resize(prev_layer->dl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size))));
		}

		for (int batch_num = 0; batch_num < prev_layer->dl_forward_output.size(); batch_num++) {
			for (int neuron_num = 0; neuron_num < prev_layer->dl_forward_output[0].size(); neuron_num++) {
				cl_forward_output[batch_num][neuron_num / (input_size * input_size)][(neuron_num / input_size) % input_size][input_size% output_size] = prev_layer->dl_forward_output[batch_num][neuron_num];
			}
		}
	}

	forward(prev_layer->cl_forward_output);
}


void convolutional_layer::rectified_linear_activation_function() {
	
	relu_flag = true;

	for (int batch_num = 0; batch_num < cl_forward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < kernals; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					cl_forward_output[batch_num][channel_num][y][x] = (cl_forward_output[batch_num][channel_num][y][x] > 0.0) ? cl_forward_output[batch_num][channel_num][y][x] : 0.0;
				}
			}
		}
	}
}

void convolutional_layer::sigmoid_activation_function() {
	
	sigmoid_flag = true;

	for (int batch_num = 0; batch_num < cl_forward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < kernals; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					cl_forward_output[batch_num][channel_num][y][x] = 1.0 / (1.0 + std::pow(e, -cl_forward_output[batch_num][channel_num][y][x]));
				}
			}
		}
	}
}

double convolutional_layer::loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	double result = 0.0;
	double batch_size = double(batched_targets.size());
	
	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {
		for (int channel_num = 0; channel_num < kernals; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					result += ((cl_forward_output[batch_num][channel_num][y][x] - batched_targets[batch_num][channel_num][y][x]) * (cl_forward_output[batch_num][channel_num][y][x] - batched_targets[batch_num][channel_num][y][x])) / batch_size;
				}
			}
		}
	}

	return result;
}

void convolutional_layer::init_backpropigation(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
	
	if (cl_backward_output.size() != cl_forward_output.size()) {
		cl_backward_output.resize(cl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(kernals, std::vector<std::vector<double>>(output_size, std::vector<double>(output_size))));
	}
	
	int output_y_idx = 0;
	int output_x_idx = 0;
	
	for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < kernals; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					cl_backward_output[batch_num][channel_num][y][x] = 2 * (cl_forward_output[batch_num][channel_num][y][x] - batched_targets[batch_num][channel_num][y][x]) * (1.0 / double(batched_targets.size()));
					
					if (sigmoid_flag) {
						cl_backward_output[batch_num][channel_num][y][x] *= (cl_forward_output[batch_num][channel_num][y][x] * (1 - cl_forward_output[batch_num][channel_num][y][x]));
					}
					else if (relu_flag && cl_forward_output[batch_num][channel_num][y][x] == 0.0) {
						cl_backward_output[batch_num][channel_num][y][x] = 0.0;
					}
				}
			}
		}
	}

	

	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < kernal_size; y++) {
				for (int x = 0; x < kernal_size; x++) {
					d_weights[kernal_num][channel_num][y][x] = 0.0;
				}
			}
		}
		d_bais[kernal_num] = 0.0;
	}

	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		output_y_idx = 0;
		for (int y = 0 - padding; y < input_size - kernal_size + 1 + padding; y += stride) {
			output_x_idx = 0;
			for (int x = 0 - padding; x < input_size - kernal_size + 1 + padding; x += stride) {

				for (int a = 0; a < kernal_size; a++) {

					if (y + a < 0 || y + a >= input_size) {
						continue;
					}

					for (int b = 0; b < kernal_size; b++) {

						if (x + b < 0 || x + b >= input_size) {
							continue;
						}

						for (int channel_num = 0; channel_num < input_depth; channel_num++) {
							for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
								d_weights[kernal_num][channel_num][a][b] += (cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * batched_inputs[batch_num][channel_num][y + a][x + b]);
							}
						}
					}
				}
				output_x_idx++;
			}
			output_y_idx++;
		}
	}

	for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
		for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					d_bais[kernal_num] += cl_backward_output[batch_num][kernal_num][y][x];
				}
			}
		}
	}

}

void convolutional_layer::init_backpropigation(layer* prev_layer, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	if (prev_layer->dl_flag) {

		if (prev_layer->cl_forward_output.size() != prev_layer->dl_forward_output.size()) {
			prev_layer->cl_forward_output.resize(prev_layer->dl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size))));
		}

		for (int batch_num = 0; batch_num < prev_layer->dl_forward_output.size(); batch_num++) {
			for (int neuron_num = 0; neuron_num < prev_layer->dl_forward_output[0].size(); neuron_num++) {
				cl_forward_output[batch_num][neuron_num / (input_size * input_size)][(neuron_num / input_size) % input_size][input_size % output_size] = prev_layer->dl_forward_output[batch_num][neuron_num];
			}
		}
	}

	init_backpropigation(prev_layer->cl_forward_output, batched_targets);

	int output_y_idx = 0;
	int output_x_idx = 0;

	if (prev_layer->cl_backward_output.size() != batched_targets.size()) {
		prev_layer->cl_backward_output.resize(batched_targets.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size, 0))));
	}
	else {
		for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
			for (int channel_num = 0; channel_num < input_depth; channel_num++) {
				for (int y = 0; y < input_size; y++) {
					for (int x = 0; x < input_size; x++) {
						prev_layer->cl_backward_output[batch_num][channel_num][y][x] = 0.0;
					}
				}
			}
		}
	}

	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
		
		output_y_idx = 0;
		for (int y = 0 - padding; y < input_size - kernal_size + 1 + padding; y += stride) {
			output_x_idx = 0;
			for (int x = 0 - padding; x < input_size - kernal_size + 1 + padding; x += stride) {

				for (int a = 0; a < kernal_size; a++) {

					if (y + a < 0 || y + a >= input_size) {
						continue;
					}

					for (int b = 0; b < kernal_size; b++) {

						if (x + b < 0 || x + b >= input_size) {
							continue;
						}

						for (int channel_num = 0; channel_num < input_depth; channel_num++) {

							for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {

								if (prev_layer->relu_flag && prev_layer->cl_forward_output[batch_num][kernal_num][y + a][y + b] == 0.0) {
									continue;
								}
								else if (prev_layer->sigmoid_flag) {
									prev_layer->cl_backward_output[batch_num][channel_num][y + a][x + b] += (cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * weights[kernal_num][channel_num][a][b]) * (prev_layer->cl_forward_output[batch_num][channel_num][y + a][x + b] * (1 - prev_layer->cl_forward_output[batch_num][channel_num][y + a][x + b]));
								}
								else {
									prev_layer->cl_backward_output[batch_num][channel_num][y + a][x + b] += cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * weights[kernal_num][channel_num][a][b];
								}
							}
						}
					}
				}
				output_x_idx++;
			}
			output_y_idx++;
		}
	}
	
	//need a flattening equation here. 

}


void convolutional_layer::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	int output_y_idx = 0;
	int output_x_idx = 0;

	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < kernal_size; y++) {
				for (int x = 0; x < kernal_size; x++) {
					d_weights[kernal_num][channel_num][y][x] = 0.0;
				}
			}
		}
		d_bais[kernal_num] = 0.0;
	}
	
	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		output_y_idx = 0;
		for (int y = 0 - padding; y < input_size - kernal_size + 1 + padding; y += stride) {
			output_x_idx = 0;
			for (int x = 0 - padding; x < input_size - kernal_size + 1 + padding; x += stride) {

				for (int a = 0; a < kernal_size; a++) {

					if (y + a < 0 || y + a >= input_size) {
						continue;
					}

					for (int b = 0; b < kernal_size; b++) {

						if (x + b < 0 || x + b >= input_size) {
							continue;
						}

						for (int channel_num = 0; channel_num < input_depth; channel_num++) {
							for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
								d_weights[kernal_num][channel_num][a][b] += (cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * batched_inputs[batch_num][channel_num][y + a][x + b]);
							}
						}
					}
				}
				output_x_idx++;
			}
			output_y_idx++;
		}
	}
	
	for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
		for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					d_bais[kernal_num] += cl_backward_output[batch_num][kernal_num][y][x];
				}
			}
		}
	}
}

void convolutional_layer::backward(layer* prev_layer) {

	if (prev_layer->dl_flag) {

		if (prev_layer->cl_forward_output.size() != prev_layer->dl_forward_output.size()) {
			prev_layer->cl_forward_output.resize(prev_layer->dl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size))));
		}

		for (int batch_num = 0; batch_num < prev_layer->dl_forward_output.size(); batch_num++) {
			for (int neuron_num = 0; neuron_num < prev_layer->dl_forward_output[0].size(); neuron_num++) {
				cl_forward_output[batch_num][neuron_num / (input_size * input_size)][(neuron_num / input_size) % input_size][input_size % output_size] = prev_layer->dl_forward_output[batch_num][neuron_num];
			}
		}
	}

	backward(prev_layer->cl_forward_output);

	int output_y_idx = 0;
	int output_x_idx = 0;

	if (prev_layer->cl_backward_output.size() != cl_backward_output.size()) {
		prev_layer->cl_backward_output.resize(cl_backward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size, 0))));
	}
	else {
		for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
			for (int channel_num = 0; channel_num < input_depth; channel_num++) {
				for (int y = 0; y < input_size; y++) {
					for (int x = 0; x < input_size; x++) {
						prev_layer->cl_backward_output[batch_num][channel_num][y][x] = 0.0;
					}
				}
			}
		}
	}

	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {

		output_y_idx = 0;
		for (int y = 0 - padding; y < input_size - kernal_size + 1 + padding; y += stride) {
			output_x_idx = 0;
			for (int x = 0 - padding; x < input_size - kernal_size + 1 + padding; x += stride) {

				for (int a = 0; a < kernal_size; a++) {

					if (y + a < 0 || y + a >= input_size) {
						continue;
					}

					for (int b = 0; b < kernal_size; b++) {

						if (x + b < 0 || x + b >= input_size) {
							continue;
						}

						for (int channel_num = 0; channel_num < input_depth; channel_num++) {

							for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {

								if (prev_layer->relu_flag && prev_layer->cl_forward_output[batch_num][kernal_num][y + a][y + b] == 0.0) {
									continue;
								}
								else if (prev_layer->sigmoid_flag) {
									prev_layer->cl_backward_output[batch_num][channel_num][y + a][x + b] += (cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * weights[kernal_num][channel_num][a][b]) * (prev_layer->cl_forward_output[batch_num][channel_num][y + a][x + b] * (1 - prev_layer->cl_forward_output[batch_num][channel_num][y + a][x + b]));
								}
								else {
									prev_layer->cl_backward_output[batch_num][channel_num][y + a][x + b] += cl_backward_output[batch_num][kernal_num][output_y_idx][output_x_idx] * weights[kernal_num][channel_num][a][b];
								}
							}
						}
					}
				}
				output_x_idx++;
			}
			output_y_idx++;
		}
	}
	
	//flatten if needed.
}

void convolutional_layer::update_parameters(double learning_rate) {

	double paramater_update = 0.0;
	for (int kernal_num = 0; kernal_num < kernals; kernal_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < kernal_size; y++) {
				for (int x = 0; x < kernal_size; x++) {
					paramater_update = (sdg_mass * sgd_mass_weights[kernal_num][channel_num][y][x]) - (d_weights[kernal_num][channel_num][y][x] * learning_rate);
					weights[kernal_num][channel_num][y][x] += paramater_update;
					sgd_mass_weights[kernal_num][channel_num][y][x] = paramater_update;
				}
			}
		}
		paramater_update = (sdg_mass * sgd_mass_bais[kernal_num]) - (d_bais[kernal_num] * learning_rate);
		bais[kernal_num] += paramater_update;
		sgd_mass_bais[kernal_num] = paramater_update;
	}
}

std::vector<int> convolutional_layer::get_shape() {
	return { input_size, input_depth, kernals, kernal_size, padding, stride };
}

std::vector<std::vector<std::vector<std::vector<double>>>> convolutional_layer::get_cl_weights() {
	return weights;
}
std::vector<double> convolutional_layer::get_bais() {
	return bais;
}
