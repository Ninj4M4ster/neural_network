#include "Core.h"
#include <fstream>
#include <math.h>

namespace core {

Core::Core() {
  weights_.at(0) = std::vector<std::vector<double>>(
      4,
      std::vector<double>(2)
      );
  weights_.at(1) = std::vector<std::vector<double>>(
      1,
      std::vector<double>(4)
      );
  offsets_.at(0) = std::vector<double>(4);
  offsets_.at(1) = std::vector<double>(1);
  loadWeights();
}

void Core::loadWeights() {
  std::fstream f;
  f.open(kFileName);
  std::string input;

  // index of weights and offsets vectors
  int core_index = 0;
  // read file line by line
  while(std::getline(f, input)) {
    std::vector<double> data;
    std::string val;

    // read numbers and push to vector if a space is found
    for(char c : input) {
      if(c == ' ') {
        data.push_back(std::stod(val));
        val = "";
      } else {
        val += c;
      }
    }
    data.push_back(std::stod(val));

    // load weights and offsets
    int perceptron_index = 0;
    for(int i = 0; i < data.size(); i += weights_.at(core_index).at(0).size() + 1) {
      for(int j = 0; j < weights_.at(core_index).at(0).size(); j++) {
        weights_.at(core_index).at(perceptron_index).at(j) = data.at(i + j);
      }
      offsets_.at(core_index).at(perceptron_index) = data.at(i + weights_.at(core_index).at(0).size());
      perceptron_index++;
    }
    core_index++;
  }
  f.close();
}

void Core::saveWeights() {
  std::fstream f;
  f.open(kFileName);

  // input data
  for(int i = 0; i < weights_.size(); i++) {
    for(int j = 0; j < weights_.at(i).size(); j++) {
      for(double weight: weights_.at(i).at(j)) {
        f << weight << " ";
      }
      f << offsets_.at(i).at(j);
      if(j != weights_.at(i).size() - 1)
        f << " ";
    }
    if(i != weights_.size() - 1)
      f << "\n";
  }

  f.close();
}

void Core::chooseRelu() {
  activation_function_ = ActivationFunctions::ReLU;
}

void Core::chooseSigmoid() {
  activation_function_ = ActivationFunctions::sigmoid;
}

void Core::chooseNoneNormalization() {
  normalization_ = Normalization::none;
}

void Core::chooseL1Normalization() {
  normalization_ = Normalization::L1;
}

void Core::chooseL2Normalization() {
  normalization_ = Normalization::L2;
}

double Core::Sigmoid(double x) {
  static constexpr double e = 2.71828182845904523536;
  return 1.0 / (1.0 + std::pow(e, -1.0 * x));
}

double Core::Relu(double x) {
  return std::max(0.0, x);
}

void Core::Normalize(double & x1, double & x2) {
  switch(normalization_) {
    case Normalization::none: {
      break;
    } case Normalization::L1: {
      double sum = x1 + x2;
      x1 = x1 / sum;
      x2 = x2 / sum;
      break;
    } case Normalization::L2: {
      double sum = std::pow(M_E, x1) + std::pow(M_E, x2);
      x1 = std::pow(M_E, x1) / sum;
      x2 = std::pow(M_E, x2) / sum;
      break;
    } default: {
      break;
    }
  }
}

double Core::Activate(double x) {
  switch (activation_function_) {
    case ActivationFunctions::ReLU: {
      return Relu(x);
    } case ActivationFunctions::sigmoid: {
        return Sigmoid(x);
    } default: {
      return Relu(x);
    }
  }
}

double Core::Derivative(double x) {
  switch (activation_function_) {
    case ActivationFunctions::ReLU: {
      if(x < 0.0)
        return 0.0;
      return 1.0;
    } case ActivationFunctions::sigmoid: {
      return Sigmoid(x) * (1.0 - Sigmoid(x));
    } default: {
      if(x < 0.0)
        return 0.0;
      return 1.0;
    }
  }
}

void Core::setLearningCoefficient(double x) {
  learning_coefficient_ = x;
}

void Core::train(std::vector<std::pair<double, double>> & data, std::vector<int> & labels) {
  for(int i = 0; i < data.size(); i++) {
    // prediction phase
    double x1 = data.at(i).first;
    double x2 = data.at(i).second;

    Normalize(x1, x2);

    std::vector<double> hidden_layers_results = std::vector<double>();
    for(int j = 0; j < weights_.at(0).size(); j++) {
      double val = weights_.at(0).at(j).at(0) * x1 +
          weights_.at(0).at(j).at(1) * x2 +
          offsets_.at(0).at(j);
      hidden_layers_results.push_back(val);
    }

    double output = 0.0;
    for(int j = 0; j < hidden_layers_results.size(); j++) {
      output += weights_.at(1).at(0).at(j) * Activate(hidden_layers_results.at(j));
    }

    output = output + offsets_.at(1).at(0);
    int answer = Activate(output) > 0.5 ? 1 : 0;

    // weights correction phase

    // calculate new weights for output layer
    double output_error = Derivative(output) * ((double)labels.at(i) - (double) answer);
    std::vector<double> hidden_to_output_errors;
    for(int j = 0; j < weights_.at(1).at(0).size(); j++) {
      hidden_to_output_errors.push_back(learning_coefficient_ * output_error * Activate(hidden_layers_results.at(j)));
    }

    // calculate new weights for hidden layer
    std::vector<double> input_vector = std::vector<double>{x1, x2};
    std::vector<std::vector<double>> input_to_hidden_errors_corrections;
    std::vector<double> input_to_hidden_errors;
    for(int j = 0; j < weights_.at(0).size(); j++) {
      input_to_hidden_errors_corrections.push_back(std::vector<double>());
      double error =
          Derivative(hidden_layers_results.at(j)) * weights_.at(1).at(0).at(j) * output_error;
      input_to_hidden_errors.push_back(error);
      for(int k = 0; k < weights_.at(0).at(j).size(); k++) {
        input_to_hidden_errors_corrections.at(j).push_back(learning_coefficient_ * input_vector.at(k) * error);
      }
    }

    // update weights
    for(int j = 0; j < weights_.at(0).size(); j++) {
      for(int k = 0; k < weights_.at(0).at(j).size(); k++) {
        weights_.at(0).at(j).at(k) += input_to_hidden_errors_corrections.at(j).at(k);
      }
    }

    for(int k = 0; k < weights_.at(1).at(0).size(); k++) {
      weights_.at(1).at(0).at(k) += hidden_to_output_errors.at(k);
    }

    // update bias
    for(int j = 0; j < offsets_.at(0).size(); j++) {
      offsets_.at(0).at(j) += learning_coefficient_ * input_to_hidden_errors.at(j);
    }

    offsets_.at(1).at(0) += learning_coefficient_ * output_error;
  }
}

int Core::predict(double x1, double x2) {
  Normalize(x1, x2);
  std::vector<double> hidden_layers_results = std::vector<double>();
  for(int i = 0; i < weights_.at(0).size(); i++) {
    double val = weights_.at(0).at(i).at(0) * x1 +
        weights_.at(0).at(i).at(1) * x2 +
        offsets_.at(0).at(i);
    hidden_layers_results.push_back(Activate(val));
  }

  double output = 0.0;
  for(int i = 0; i < hidden_layers_results.size(); i++) {
    output += weights_.at(1).at(0).at(i) * hidden_layers_results.at(i);
  }

  return Activate(output + offsets_.at(1).at(0)) > 0.5 ? 1 : 0;
}

double Core::evaluate(std::vector<std::pair<double, double>> & data, std::vector<int> & labels) {
  int correct = 0;
  for(int i = 0; i < data.size(); i++) {
    auto pair = data.at(i);
    int result = predict(pair.first, pair.second);
    if(result == labels.at(i))
      correct++;
  }
  return (double) correct / (double) data.size();
}


}  // namespace core
