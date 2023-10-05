#include "model.h"
#include <iostream>

void model::init_model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass) {
    type = _model_type;
    starting_learning_rate = _learning_rate;
    learning_rate = _learning_rate;
    decay_rate = _decay_rate;
    sgd_mass = _sgd_mass;
    layer_count = 0;
    step = 0;
}

model::model(int _model_type, double _learning_rate, double _decay_rate, double _sgd_mass) {
    init_model(_model_type, _learning_rate, _decay_rate, _sgd_mass);
}
model::model(int _model_type, double _learning_rate) {
    init_model(_model_type, learning_rate, 0, 0.9);
}
model::model(int _model_type) {
    init_model(_model_type, 1e-3, 0, 0.9);
}
model::model() {
    init_model(model_type::regression, 1e-3, 0, 0.9);
}

void model::add_dense_layer(int _inputs, int _neurons, int _activation_function) {
    layers.push_back(new dense_layer(_inputs, _neurons, sgd_mass));
    activation_functions.push_back(_activation_function);
    layer_count++;
}

void model::add_convolutional_layer(int _input_size, int _input_channels, int _kernals, int _kernal_size, int _padding, int _stride, int _activation_function){
    layers.push_back(new convolutional_layer(_input_size, _input_channels, _kernals, _kernal_size, _padding, _stride, sgd_mass));
    activation_functions.push_back(_activation_function);
    layer_count++;
}

void model::add_pooling_layer(int _input_size, int _input_channels, int _kernal_size, int _padding, int _stride) {
    layers.push_back(new max_pooling_layer(_input_size, _input_channels, _kernal_size, _stride));
    activation_functions.push_back(activation_function::linear);
    layer_count++;
}


void model::forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, int starting_layer, int ending_layer) {
    layers[starting_layer]->forward(batched_inputs);
    if (activation_functions[starting_layer] == activation_function::sigmoid) {
        layers[starting_layer]->sigmoid_activation_function();
    }
    else if (activation_functions[starting_layer] == activation_function::rectified_linear) {
        layers[starting_layer]->rectified_linear_activation_function();
    }
    else if (activation_functions[starting_layer] == activation_function::softmax) {
        layers[starting_layer]->softmax_activation_function();
    }

    for (int i = starting_layer + 1; i < layers.size() && i <= ending_layer; i++) {
        layers[i]->forward(layers[i - 1]);
        if (activation_functions[i] == activation_function::linear) {
            continue;
        }
        else if (activation_functions[i] == activation_function::sigmoid) {
            layers[i]->sigmoid_activation_function();
        }
        else if (activation_functions[i] == activation_function::rectified_linear) {
            layers[i]->rectified_linear_activation_function();
        }
        else if (activation_functions[i] == activation_function::softmax) {
            layers[i]->softmax_activation_function();
        }
    }
}
void model::forward(std::vector<std::vector<double>>& batched_inputs, int starting_layer, int ending_layer) {
    layers[starting_layer]->forward(batched_inputs);
    if (activation_functions[starting_layer] == activation_function::sigmoid) {
        layers[starting_layer]->sigmoid_activation_function();
    }
    else if (activation_functions[starting_layer] == activation_function::rectified_linear) {
        layers[starting_layer]->rectified_linear_activation_function();
    }
    else if (activation_functions[starting_layer] == activation_function::softmax) {
        layers[starting_layer]->softmax_activation_function();
    }

    for (int i = starting_layer + 1; i < layers.size() && i <= ending_layer; i++) {
        layers[i]->forward(layers[i - 1]);
        if (activation_functions[i] == activation_function::linear) {
            continue;
        }
        else if (activation_functions[i] == activation_function::sigmoid) {
            layers[i]->sigmoid_activation_function();
        }
        else if (activation_functions[i] == activation_function::rectified_linear) {
            layers[i]->rectified_linear_activation_function();
        }
        else if (activation_functions[i] == activation_function::softmax) {
            layers[i]->softmax_activation_function();
        }
    }
}
void model::forward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
    forward(batched_inputs, 0, layer_count - 1);
}
void model::forward(std::vector<std::vector<double>>& batched_inputs) {
    forward(batched_inputs, 0, layer_count - 1);
}

std::vector<std::vector<double>> model::dense_layer_output(int layer_index) {
    return layers[layer_index]->dl_forward_output;
}
std::vector<std::vector<std::vector<std::vector<double>>>> model::convolutional_layer_output(int layer_index) {
    return layers[layer_index]->cl_forward_output;
}

double model::loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
    return layers.back()->loss(batched_targets);
}
double model::loss(std::vector<std::vector<double>>& batched_targets) {
    return layers.back()->loss(batched_targets);
}
double model::loss(std::vector<int>&batched_targets) {
    return layers.back()->loss(batched_targets);
}

void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets, int starting_layer, int ending_layer) {

    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        layers[0]->init_backpropigation(batched_inputs, batched_targets);
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer-1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[(0 > starting_layer) ? 0 : starting_layer]->backward(batched_inputs);
}
void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets, int starting_layer, int ending_layer) {
    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        //error
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer - 1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[(0 > starting_layer) ? 0 : starting_layer]->backward(batched_inputs);
}
void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets, int starting_layer, int ending_layer) {
    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        //error
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer - 1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[0]->backward(batched_inputs);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets, int starting_layer, int ending_layer) {
    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        //error
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer - 1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[(0 > starting_layer) ? 0 : starting_layer]->backward(batched_inputs);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets, int starting_layer, int ending_layer) {
    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        layers[0]->init_backpropigation(batched_inputs, batched_targets);
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer - 1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[(0 > starting_layer) ? 0 : starting_layer]->backward(batched_inputs);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets, int starting_layer, int ending_layer) {
    ending_layer = (ending_layer < layer_count - 1) ? ending_layer : layer_count - 1;

    if (ending_layer == 0) {
        layers[0]->init_backpropigation(batched_inputs, batched_targets);
        return;
    }
    else {
        layers[ending_layer]->init_backpropigation(layers[ending_layer - 1], batched_targets);
    }

    for (int i = ending_layer - 1; i >= 1 && i >= starting_layer + 1; i--) {
        layers[i]->backward(layers[i - 1]);
    }

    layers[(0 > starting_layer) ? 0 : starting_layer]->backward(batched_inputs);
}

void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}
void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}
void model::backward(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}
void model::backward(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets) {
    backward(batched_inputs, batched_targets, 0, layer_count - 1);
}

void model::update_parameters(int starting_layer, int ending_layer) {
    for (int i = starting_layer; i < layer_count && i <= ending_layer; i++) {
        layers[i]->update_parameters(learning_rate);
    }
}
void model::update_parameters() {
    update_parameters(0, layer_count-1);
}

void model::train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}
void model::train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}
void model::train(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs, std::vector<int>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}
void model::train(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}
void model::train(std::vector<std::vector<double>>& batched_inputs, std::vector<std::vector<double>>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}
void model::train(std::vector<std::vector<double>>& batched_inputs, std::vector<int>& batched_targets) {
    forward(batched_inputs);
    backward(batched_inputs, batched_targets);
    update_parameters();
}

void model::decay_learning_rate() {
    step++;
    learning_rate = starting_learning_rate * (1.0 / (1.0 + decay_rate * step));
}

void model::save_model(const std::string& file_name) {

    std::ofstream file(file_name, std::ios::out);
    if (!file.is_open()) {
        std::cout << "could not open file\n";
        return;
    }
    for (int i = 0; i < layers.size(); i++) {

        if (layers[i]->dl_flag) {
            write_dense_layer(file, i);
        }
        else if(layers[i]->pl_flag) {
            write_pooling_layer(file, i);
        }
        else {
            write_convolutional_layer(file, i);
        }
    }
    file.close();
}

void model::write_dense_layer(std::ofstream& file, int layer_idx) {
    std::vector<int> shape = layers[layer_idx]->get_shape();
    std::vector<std::vector<double>> weights = layers[layer_idx]->get_dl_weights();
    std::vector<double> bais = layers[layer_idx]->get_bais();

    file << "dense_layer,"<<shape[0]<<','<<shape[1]<<','<<activation_functions[layer_idx] << ",\n";

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            file << weights[i][j] << ",";
        }
        file << "\n";
    }

    for (int i = 0; i < bais.size(); i++) {
        file << bais[i] << ",";
    }
    file << "\n";
}
void model::write_convolutional_layer(std::ofstream& file, int layer_idx) {

    std::vector<int> shape = layers[layer_idx]->get_shape();
    std::vector<std::vector<std::vector<std::vector<double>>>> weights = layers[layer_idx]->get_cl_weights();
    std::vector<double> bais = layers[layer_idx]->get_bais();

    file << "convolutional_layer,";
    for (int i = 0; i < shape.size(); i++) {
        file << shape[i] << ",";
    }
    file << activation_functions[layer_idx] << ",\n";

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            for (int y = 0; y < weights[0][0].size(); y++) {
                for (int x = 0; x < weights[0][0][0].size(); x++) {
                    file << weights[i][j][y][x] << ",";
                }
            }
            file << "\n";
        }
    }

    for (int i = 0; i < bais.size(); i++) {
        file << bais[i] << ",";
    }
    file<< "\n";

}
void model::write_pooling_layer(std::ofstream& file, int layer_idx) {
    std::vector<int> shape = layers[layer_idx]->get_shape();

    file << "max_pooling_layer,";
    for (int i = 0; i < shape.size(); i++) {
        file << shape[i] << ",";
    }
    file << "\n";
}

void model::load_model(const std::string& file_name) {

    std::ifstream file(file_name);
    std::string current_line;
    std::string parsed_data;
    std::vector<int> shape;


    std::vector<double> bais;
    int idx = 0;
    int line_idx = 0;

    if (!file.is_open()) {
        std::cout << "could not open file\n";
        return;
    }

    while (std::getline(file, current_line)) {

        if (current_line.substr(0, 12) == "dense_layer,") {
            std::vector<std::vector<double>> dl_weights;
            idx = 0;
            shape.resize(3);
            for (int i = 12; i < current_line.length(); i++) {

                if (current_line[i] != ',' && current_line[i] != '\n') {
                    parsed_data += current_line[i];
                }
                else if (!parsed_data.empty()) {
                    shape[idx] = std::stoi(parsed_data);
                    parsed_data.clear();
                    idx++;
                }
            }
            if (!parsed_data.empty()) {
                shape[idx] = std::stoi(parsed_data);
                parsed_data.clear();
            }
            dl_weights.resize(shape[1], std::vector<double>(shape[0]));

            idx = 0;
            for (int i = 0; i < shape[1]; i++) {
                std::getline(file, current_line);
                for (int j = 0; j < current_line.length(); j++) {
                    if (current_line[j] != ',' && current_line[j] != '\n') {
                        parsed_data += current_line[j];
                    }
                    else if (!parsed_data.empty()) {
                        dl_weights[idx / shape[0]][idx % shape[0]] = std::stod(parsed_data);
                        parsed_data.clear();
                        idx++;
                    }
                }
                if (!parsed_data.empty()) {
                    dl_weights[idx / shape[0]][idx % shape[0]] = std::stod(parsed_data);
                    parsed_data.clear();
                }
            }
            std::getline(file, current_line);
            idx = 0;
            bais.resize(shape[1]);

            for (int i = 0; i < current_line.length(); i++) {

                if (current_line[i] != ',' && current_line[i] != '\n') {
                    parsed_data += current_line[i];
                }
                else if (!parsed_data.empty()) {
                    bais[idx] = std::stod(parsed_data);
                    parsed_data.clear();
                    idx++;
                }
            }
            if (!parsed_data.empty()) {
                bais[idx] = std::stod(parsed_data);
                parsed_data.clear();
            }
            dense_layer* new_layer = new dense_layer(shape[0], shape[1], sgd_mass);
            new_layer->weights = dl_weights;
            new_layer->bais = bais;

            layers.push_back(new_layer);
            activation_functions.push_back(shape[2]);
            dl_weights.clear();
            bais.clear();
            layer_count++;
        }
        else if (current_line.substr(0, 20) == "convolutional_layer,") {

            std::vector<std::vector<std::vector<std::vector<double>>>> cl_weights;
            idx = 0;
            shape.resize(7);
            for (int i = 20; i < current_line.length(); i++) {

                if (current_line[i] != ',' && current_line[i] != '\n') {
                    parsed_data += current_line[i];
                }
                else if (!parsed_data.empty()) {
                    shape[idx] = std::stoi(parsed_data);
                    parsed_data.clear();
                    idx++;
                }
            }
            if (!parsed_data.empty()) {
                shape[idx] = std::stoi(parsed_data);
                parsed_data.clear();
            }

            idx = 0;

            cl_weights.resize(shape[2], std::vector<std::vector<std::vector<double>>>(shape[1], std::vector<std::vector<double>>(shape[3], std::vector<double>(shape[3]))));

            for (int i = 0; i < shape[2]; i++) {

                for (int j = 0; j < shape[1]; j++) {
                    std::getline(file, current_line);
                    int weight_idx = 0;
                    for (int z = 0; z < current_line.length(); z++) {

                        if (current_line[z] != ',' && current_line[z] != '\n') {
                            parsed_data += current_line[z];
                        }
                        else if (!parsed_data.empty()) {
                            cl_weights[i][j][weight_idx / shape[3]][weight_idx % shape[3]] = std::stod(parsed_data);
                            parsed_data.clear();
                            weight_idx++;
                        }
                    }

                    if (!parsed_data.empty()) {
                        cl_weights[i][j][weight_idx / shape[3]][weight_idx % shape[3]] = std::stod(parsed_data);
                        parsed_data.clear();
                        weight_idx++;
                    }

                }

            }
            idx = 0;
            bais.resize(shape[2]);
            std::getline(file, current_line);
            for (int i = 0; i < current_line.length(); i++) {

                if (current_line[i] != ',' && current_line[i] != '\n') {
                    parsed_data += current_line[i];
                }
                else if (!parsed_data.empty()) {
                    bais[idx] = std::stod(parsed_data);
                    parsed_data.clear();
                    idx++;
                }
            }
            if (!parsed_data.empty()) {
                bais[idx] = std::stod(parsed_data);
                parsed_data.clear();
            }
            convolutional_layer* new_layer = new convolutional_layer(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], sgd_mass);
            new_layer->weights = cl_weights;
            new_layer->bais = bais;
            layers.push_back(new_layer);
            activation_functions.push_back(shape[6]);
            layer_count++;
        }
        else if (current_line.substr(0, 18) == "max_pooling_layer,") {
            shape.resize(4);
            idx = 0;
            for (int i = 18; i < current_line.length(); i++) {

                if (current_line[i] != ',' && current_line[i] != '\n') {
                    parsed_data += current_line[i];
                }
                else if (!parsed_data.empty()) {
                    shape[idx] = std::stoi(parsed_data);
                    parsed_data.clear();
                    idx++;
                }
            }

            if (!parsed_data.empty()) {
                shape[idx] = std::stoi(parsed_data);
                parsed_data.clear();
            }

            max_pooling_layer* new_layer = new max_pooling_layer(shape[0], shape[1], shape[2], shape[3]);
            layers.push_back(new_layer);
            activation_functions.push_back(activation_function::linear);
            layer_count++;
        }
    }

}

model::~model() {
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}
