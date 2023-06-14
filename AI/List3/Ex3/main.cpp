#include "generator.h"
#include "Core.cpp"
#include <memory>

int main() {
  std::shared_ptr<core::Core> network = std::make_shared<core::Core>();

  std::vector<std::pair<double, double>> data;
  std::vector<int> labels;


  network->setLearningCoefficient(0.01);
  network->chooseSigmoid();
//  network->chooseL1Normalization();
  network->chooseL2Normalization();

  generate_data(data, labels, 10000);
  double result = network->evaluate(data, labels);
  std::cout << "Prediction success rate: " << result << std::endl;



  std::cout << "Prediction: " << network->predict(-0.9, -0.9) << std::endl;

  data.clear();
  labels.clear();
  generate_data(data, labels, 1000000);

  network->train(data, labels);
  result = network->evaluate(data, labels);
  std::cout << "Prediction success rate: " << result << std::endl;


  data.clear();
  labels.clear();
  generate_data(data, labels, 10000);
  result = network->evaluate(data, labels);
  std::cout << "Prediction success rate: " << result << std::endl;

  std::cout << "Prediction: " << network->predict(-0.9, -0.9) << std::endl;

  network->saveWeights();

}