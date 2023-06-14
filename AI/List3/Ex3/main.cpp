#include "generator.h"
#include "Core.cpp"
#include <memory>

int main() {
  std::shared_ptr<core::Core> network = std::make_shared<core::Core>();

  std::vector<std::pair<double, double>> data;
  std::vector<int> labels;

  network->setLearningCoefficient(0.01);
  network->chooseSigmoid();
  network->chooseL1Normalization();

//  generate_data(data, labels, 1000000);
//
//  network->train(data, labels);
//  double result = network->evaluate(data, labels);
//  std::cout << "Prediction success rate: " << result << std::endl;

  std::cout << "Prediction: " << network->predict(0.8, 0.8);

  network->saveWeights();

}