#include "layer.h"
#include <iostream>

max_pooling_layer::max_pooling_layer(int _input_size, int _input_depth, int _kernal_size, int _stride) {

	input_size = _input_size;
	input_depth = _input_depth;
	kernal_size = _kernal_size;
	stride = _stride;

	cl_flag = true;
	dl_flag = false;
	pl_flag = true;

	relu_flag = false;
	sigmoid_flag = false;
	softmax_flag = false;

	output_size = ((input_size  - kernal_size) / stride) + 1;
}

void max_pooling_layer::forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	if (cl_forward_output.size() != batched_inputs.size()) {
		cl_forward_output.resize(batched_inputs.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(output_size, std::vector<double>(output_size))));
	}
	for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					cl_forward_output[batch_num][channel_num][y][x] = 0.0;
				}
			}
		}
	}

	double result = 0;
	int output_y_idx = 0;
	int output_x_idx = 0;

	for (int batch_num = 0; batch_num < batched_inputs.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			output_y_idx = 0;
			for (int y = 0; y < input_size - kernal_size + 1; y += stride) {
				output_x_idx = 0;
				for (int x = 0; x < input_size - kernal_size + 1; x += stride) {

					result = FLT_MIN;
					for (int a = 0; a < kernal_size; a++) {
						for (int b = 0; b < kernal_size; b++) {
							result = (result > batched_inputs[batch_num][channel_num][y + a][x + b]) ? result : batched_inputs[batch_num][channel_num][y + a][x + b];
						}
					}
					cl_forward_output[batch_num][channel_num][output_y_idx][output_x_idx] = result;
					output_x_idx++;
				}

				output_y_idx++;
			}
		}
	}
}

void max_pooling_layer::forward(layer* prev_layer) {

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

	forward(prev_layer->cl_forward_output);
}

void max_pooling_layer::backward(layer* prev_layer) {

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

	if (prev_layer->cl_backward_output.size() != cl_backward_output.size()) {
		prev_layer->cl_backward_output.resize(cl_backward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(input_size, std::vector<double>(input_size))));
	}
	
	int max_idx_y = 0;
	int max_idx_x = 0;
	int output_y_idx = 0;
	int output_x_idx = 0;

	for (int batch_num = 0; batch_num < prev_layer->cl_backward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < input_size; y++) {
				for (int x = 0; x < input_size; x++) {
					prev_layer->cl_backward_output[batch_num][channel_num][y][x] = 0;
				}
			}
		}
	}
	
	for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {

			output_y_idx = 0;
			for (int y = 0; y < input_size - kernal_size + 1; y += stride) {
				output_x_idx = 0;
				for (int x = 0; x < input_size - kernal_size + 1; x += stride) {

					for (int a = 0; a < kernal_size; a++) {

						for (int b = 0; b < kernal_size; b++) {
							if (prev_layer->cl_forward_output[batch_num][channel_num][y + a][x + b] > prev_layer->cl_forward_output[batch_num][channel_num][y + max_idx_y][x + max_idx_x]) {
								max_idx_y = a;
								max_idx_x = b;
							}
						}
					}

					prev_layer->cl_backward_output[batch_num][channel_num][y + max_idx_y][x + max_idx_x] += cl_backward_output[batch_num][channel_num][output_y_idx][output_x_idx];
					output_x_idx++;
				}
				output_y_idx++;
			}
		}
	}
}

void max_pooling_layer::init_backpropigation(layer* prev_layer, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	if (cl_backward_output.size() != cl_forward_output.size()) {
		cl_backward_output.resize(cl_forward_output.size(), std::vector<std::vector<std::vector<double>>>(input_depth, std::vector<std::vector<double>>(output_size, std::vector<double>(output_size))));
	}

	int output_y_idx = 0;
	int output_x_idx = 0;

	for (int batch_num = 0; batch_num < cl_backward_output.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
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
	
	backward(prev_layer);
}

double max_pooling_layer::loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	double result = 0.0;
	double batch_size = double(batched_targets.size());

	for (int batch_num = 0; batch_num < batched_targets.size(); batch_num++) {
		for (int channel_num = 0; channel_num < input_depth; channel_num++) {
			for (int y = 0; y < output_size; y++) {
				for (int x = 0; x < output_size; x++) {
					result += ((cl_forward_output[batch_num][channel_num][y][x] - batched_targets[batch_num][channel_num][y][x]) * (cl_forward_output[batch_num][channel_num][y][x] - batched_targets[batch_num][channel_num][y][x])) / batch_size;
				}
			}
		}
	}

	return result;
}

std::vector<int> max_pooling_layer::get_shape() {
	return { input_size, input_depth, kernal_size, stride };
}
