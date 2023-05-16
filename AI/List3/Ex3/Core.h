#ifndef WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_
#define WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_

#include <iostream>
#include <vector>

namespace core {

class Core {
 private:
  static constexpr auto kFileName = "weights.txt";
  // weights and offsets of the network
  std::vector<std::vector<double>> weights_ = std::vector<std::vector<double>>();
  std::vector<std::vector<double>> offsets_ = std::vector<std::vector<double>>();

  enum ActivationFunctions {
    ReLU,
    sigmoid
  };

  ActivationFunctions activation_function_ = ActivationFunctions::ReLU;

  enum Normalization {
    none,
    L1,
    L2
  };

  Normalization normalization_ = Normalization::none;

  // utilities
  static double Sigmoid(double x);
  static double Relu(double x);

  void saveWeights();

 public:
  // constructor
  Core();

  // utility functions
  void loadWeights();
  void chooseRelu();
  void chooseSigmoid();
  void chooseNoneNormalization();
  void chooseL1Normalization();
  void chooseL2Normalization();

  // core functions
  void train(std::vector<std::pair<double, double>> data, std::vector<double> labels);
  double predict(double x1, double x2);
  double evaluate(std::vector<std::pair<double, double>> data, std::vector<double> labels);
};

}  // namespace core

#endif  // WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_
