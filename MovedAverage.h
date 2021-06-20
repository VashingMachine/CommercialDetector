//
// Created by ktoztam on 20.06.2021.
//

#ifndef COMMERCIALDETECTOR_MOVEDAVERAGE_H
#define COMMERCIALDETECTOR_MOVEDAVERAGE_H
#include <iostream>
#include <vector>

class MovedAverage {
protected:
    std::vector<float> probabilities;
    int size;
    int iterator = 0;
public:
    MovedAverage(int size): size(size) {}

    float getAverage() {
        float sum = 0;
        for(auto &p: probabilities) {
            sum += p;
        }
        return sum / probabilities.size();
    }

    void addProb(float prob) {
        if(probabilities.size() < size) {
            probabilities.push_back(prob);
        } else {
            iterator = iterator % size;
            probabilities[iterator] = prob;
        }
        iterator++;
    }
};


#endif //COMMERCIALDETECTOR_MOVEDAVERAGE_H
