#include "layer.h"
#include <random>

const double e = 2.71828182846;

layer::layer(int _inputs, int _neurons, double _sgd_mass) {

	inputs = _inputs;
	neurons = _neurons;
	sigmoid_flag = false;
	relu_flag = false;
	sgd_mass = _sgd_mass;

	weights.resize(neurons, std::vector<double>(inputs));
	d_weights.resize(neurons, std::vector<double>(inputs, 0));
	weight_momentums.resize(neurons, std::vector<double>(inputs, 0));
	bais.resize(neurons, 0);
	d_bais.resize(neurons, 0);
	bais_momentums.resize(neurons, 0);

	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (int i = 0; i < neurons; i++) {
		for (int j = 0; j < inputs; j++) {
			weights[i][j] = distribution(generator);
		}
		bais[i] = distribution(generator);
	}

}

void layer::forward(std::vector<std::vector<double>>& batched_inputs) {
	
	if (output.size() != batched_inputs.size()) {
		output.resize(batched_inputs.size(), std::vector<double>(neurons, 0.0));
	}

	for (int x = 0; x < batched_inputs.size(); x++) {

		for (int i = 0; i < neurons; i++) {

			output[x][i] = 0.0;

			for (int j = 0; j < inputs; j++) {
				output[x][i] += (batched_inputs[x][j] * weights[i][j]);
			}
			output[x][i] += bais[i];
		}
	}

}

void layer::sigmoid_activation_function(){

	sigmoid_flag = true;
	for (int x = 0; x < output.size(); x++) {
		for (int i = 0; i < neurons; i++) {
			output[x][i] = 1 / (1 + std::pow(e, -output[x][i]));
		}
	}

}

void layer::rectified_linear_activation_function() {

	relu_flag = true;

	for (int x = 0; x < output.size(); x++) {
		for (int i = 0; i < neurons; i++) {
			output[x][i] = (output[x][i] > 0) ? output[x][i] : 0;
		}
	}
}


double layer::loss(std::vector<std::vector<double>>& batched_targets) {

	double loss = 0.0;

	for (int x = 0; x < batched_targets.size(); x++) {
		for (int i = 0; i < neurons; i++) {
			loss += (((output[x][i] - batched_targets[x][i]) * (output[x][i] - batched_targets[x][i])) / double(batched_targets.size()));
		}
	}
	
	return loss;
}

void layer::init_back_propagation(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {

	for (int x = 0; x < batched_inputs.size(); x++) {

		for (int i = 0; i < neurons; i++) {

			double last = output[x][i];

			output[x][i] = 2 * (output[x][i] - batched_targets[x][i]);

			if (sigmoid_flag) {
				output[x][i] *= (last * (1 - last));
			}
			else if(relu_flag){
				output[x][i] = (last != 0) ? output[x][i] : 0;
			}
			
			output[x][i] /= double(batched_inputs.size());
		}
	}

	for (int i = 0; i < neurons; i++) {
		for (int j = 0; j < inputs; j++) {

			d_weights[i][j] = 0;

			for (int x = 0; x < batched_inputs.size(); x++) {
				d_weights[i][j] += (output[x][i] * batched_inputs[x][j]);
			}

		}
	}

	for (int i = 0; i < neurons; i++) {

		d_bais[i] = 0;
		for (int x = 0; x < batched_inputs.size(); x++) {
			d_bais[i] += output[x][i];
		}
	}

}

void layer::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& forward_weights, std::vector<std::vector<double>>& forward_output) {

	for (int x = 0; x < batched_inputs.size(); x++) {

		for (int i = 0; i < neurons; i++) {

			double last = output[x][i];

			output[x][i] = 0;

			for (int j = 0; j < forward_weights.size(); j++) {

				output[x][i] += forward_output[x][j] * forward_weights[j][i];

			}

			if (sigmoid_flag) {
				output[x][i] *= (last * (1 - last));
			}
			else if (relu_flag) {
				output[x][i] = (last != 0) ? output[x][i] : 0;
			}
		}


	}

	for (int i = 0; i < neurons; i++) {
		for (int j = 0; j < inputs; j++) {

			d_weights[i][j] = 0;

			for (int x = 0; x < batched_inputs.size(); x++) {
				d_weights[i][j] += (output[x][i] * batched_inputs[x][j]);
			}

		}
	}

	for (int i = 0; i < neurons; i++) {

		d_bais[i] = 0;
		for (int x = 0; x < batched_inputs.size(); x++) {
			d_bais[i] += output[x][i];
		}
	}

}

void layer::train(double learning_rate) {

	double paramater_update = 0.0;
	for (int i = 0; i < neurons; i++) {
		for (int j = 0; j < inputs; j++) {

			paramater_update = (sgd_mass * weight_momentums[i][j]) - (d_weights[i][j] * learning_rate);
			weights[i][j] += paramater_update;
			weight_momentums[i][j] = paramater_update;
		}
		paramater_update = (sgd_mass * bais_momentums[i]) - (d_bais[i] * learning_rate);
		bais[i] += paramater_update;
		bais_momentums[i] = paramater_update;
	}
	
}
