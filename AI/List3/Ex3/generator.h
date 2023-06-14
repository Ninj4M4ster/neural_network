#ifndef WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_GENERATOR_H_
#define WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_GENERATOR_H_

#include <iostream>
#include <vector>
#include <random>

void generate_data(std::vector<std::pair<double, double>> & data, std::vector<int> & labels, int count) {
  std::mt19937_64 rand_gen{std::random_device{}()};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for(int i = 0; i < count; i++) {
    double x1 = dist(rand_gen);
    double x2 = dist(rand_gen);
    if(x1 * x2 >= 0) {
      labels.push_back(1);
    } else {
      labels.push_back(0);
    }
    data.push_back({x1, x2});
  }
}

#endif //WROCLAW_UNIVERSITY_OF_SCIENCE_AI_LIST3_EX3_GENERATOR_H_
