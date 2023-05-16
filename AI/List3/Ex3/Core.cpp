#include "Core.h"
#include <fstream>
#include <math.h>

namespace core {

Core::Core() {
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

    weights_.push_back(std::vector<double>());
    offsets_.push_back(std::vector<double>());
    // load weights and offsets
    for(int i = 0; i < data.size(); i += 2) {
      weights_.at(core_index).push_back(data.at(i));
      offsets_.at(core_index).push_back(data.at(i+1));
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
      f << weights_.at(i).at(j) << " " << offsets_.at(i).at(j);
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

void Core::train(std::vector<std::pair<double, double>> data, std::vector<double> labels) {

}

double Core::predict(double x1, double x2) {
  std::vector<double> hidden_layers_results = std::vector<double>();
  for(int i = 0; i < weights_.at(0).size(); i++) {
    
  }
}

double Core::evaluate(std::vector<std::pair<double, double>> data, std::vector<double> labels) {

}


}  // namespace core
