#ifndef WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_
#define WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_

#include <iostream>
#include <vector>

namespace core {

class Core {
 private:
  static constexpr auto kFileName = "weights.txt";
  // weights and offsets of the network
  std::vector<std::vector<std::vector<double>>> weights_ =
      std::vector<std::vector<std::vector<double>>>(
          2,  // hidden layer and output layer
          std::vector<std::vector<double>>()
          );
  std::vector<std::vector<double>> offsets_ =
      std::vector<std::vector<double>>(2, std::vector<double>());

  double learning_coefficient_ = 1.0;

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
  void loadWeights();

  void Normalize(double & x1, double & x2);
  double Activate(double x);
  double Derivative(double x);

 public:
  // constructor
  Core();

  // utility functions
  void chooseRelu();
  void chooseSigmoid();
  void chooseNoneNormalization();
  void chooseL1Normalization();
  void chooseL2Normalization();

  void saveWeights();

  void setLearningCoefficient(double x);

  // core functions
  void train(std::vector<std::pair<double, double>> & data, std::vector<int> & labels);
  int predict(double x1, double x2);
  double evaluate(std::vector<std::pair<double, double>> & data, std::vector<int> & labels);
};

}  // namespace core

#endif  // WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_CORE_H_
