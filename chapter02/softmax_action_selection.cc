#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "multiarmed_bandit.hh"
using namespace std;

int n_arms = 10;
multiarmed_bandit bandit(n_arms);

double softmax_action_selection(double t=2, int n_epochs=1000) {
    double score = 0.0;

    vector<double> Q(n_arms);
    vector<double> r(n_arms);
    vector<double> k(n_arms);

    for(int epoch=0; epoch<n_epochs; ++epoch) {
        double dnr = 0.0;
        for(int i=0; i<n_arms; ++i) dnr += exp(Q[i]/t);

        vector<double> p;
        for(int i=0; i<n_arms; ++i) {
            p.push_back(exp(Q[i]/t) / dnr);
        }

        std::random_device rd;
        std::mt19937 mt(rd());
        std::discrete_distribution<int> dist(p.begin(), p.end());

        int arm = dist(mt);

        double s = bandit.get_score(arm);
        score += s;
        r[arm] += s;
        k[arm]++;
        Q[arm] = r[arm] / k[arm];

        if(epoch%100==0) {
            cout << "target_arm: " << arm << endl;
        }
        if(epoch%50==0) {
            cout << score / (epoch+1) << endl;
        }
    }
    return score;
}

int main() {
    double score = softmax_action_selection();
    double best_score = bandit.best_score(1000);
    cout << "best average score: " << best_score / 1000 << endl;
    cout << "optinum degree: " << score / best_score << endl;
}
