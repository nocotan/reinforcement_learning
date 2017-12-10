#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "multiarmed_bandit.hh"
using namespace std;

int n_arms = 10;
multiarmed_bandit bandit(n_arms);

double epsilon_greedy(double eps=0.1, int n_epochs=1000, int n_explores=2) {
    double score = 0.0;

    vector<double> Q(n_arms);
    vector<double> r(n_arms);
    vector<double> k(n_arms);

    for(int i=0; i<n_explores; ++i) {
        for(int j=0; j<n_arms; ++j) {
            r[j] += bandit.get_score(j);
            k[j]++;
        }
    }

    for(int i=0; i<n_arms; ++i) Q[i] = r[i] / k[i];

    double max_value = -1;
    int target = 0;
    for(int i=0; i<n_arms; ++i) {
        if(max_value<Q[target]) {
            target = i;
            max_value = Q[target];
        }
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> random_dist(0, n_arms-1);

    for(int epoch=n_explores*n_arms; epoch<n_epochs; ++epoch) {
        int arm = 0;
        if(eps>=dist(mt)) arm = random_dist(mt);
        else arm = target;

        double s = bandit.get_score(arm);
        score += s;
        r[arm] += s;
        k[arm]++;
        Q[arm] = r[arm] / k[arm];

        if(Q[arm] > max_value) {
            target = arm;
            max_value = Q[arm];
        }
        /**
        if(epoch%100==0) {
            cout << "target_arm: " << target << endl;
        }
        **/
        if(epoch%50==0) {
            cout << score/(epoch+1) << endl;
        }
    }
    return score;
}

int main() {
    double score = epsilon_greedy();
    // bandit.print_ranges();
    double best_score = bandit.best_score(1000);
    cout << "best average score: " << best_score / 1000 << endl;
    cout << "optinum degree: " << score / best_score << endl;
}
