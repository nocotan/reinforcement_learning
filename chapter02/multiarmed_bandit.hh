#ifndef MULTIARMED_BANDIT_HH
#define MULTIARMED_BANDIT_HH

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

class multiarmed_bandit {
    private:
        int n;
        std::vector<std::uniform_real_distribution<double>> arms;
        std::vector<std::pair<double, double>> ranges;
    public:
        multiarmed_bandit(int n) : n(n) {
            std::random_device rd;
            std::mt19937 mt(rd());
            for(int i=0; i<n; ++i) {
                std::uniform_real_distribution<double> range(0.0, 10.0);
                double a = range(mt);
                double b = range(mt);
                std::uniform_real_distribution<double> dist(std::min(a, b), std::max(a, b));
                arms.push_back(dist);
                ranges.push_back(std::make_pair(std::min(a, b), std::max(a, b)));
            }
        }

        double get_score(int idx) {
            if(idx>=n) {
                std::cout << "Error: index is out of range." << std::endl;
                exit(1);
            }
            auto dist = arms[idx];
            std::random_device rd;
            std::mt19937 mt(rd());
            return dist(mt);
        }

        void print_ranges() {
            std::cout << "True range of each arms" << std::endl;
            for(int i=0; i<n; ++i) {
                std::cout << ranges[i].first << ", " << ranges[i].second << std::endl;
            }
        }
        double best_score(int n_epoch) {
            std::vector<double> scores(n);
            std::random_device rd;
            std::mt19937 mt(rd());
            for(int i=0; i<n; ++i) {
                auto dist = arms[i];
                for(int epoch=0; epoch<n_epoch; ++epoch) {
                    scores[i] += dist(mt);
                }
            }
            std::sort(scores.begin(), scores.end());
            return scores[scores.size()-1];
        }
};

#endif
