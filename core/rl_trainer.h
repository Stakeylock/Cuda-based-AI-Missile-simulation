#ifndef RL_TRAINER_H
#define RL_TRAINER_H

#include "../core/config.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

using Eigen::MatrixXf;
using Eigen::VectorXf;

struct StrategistNetwork;
struct PilotNetwork;
class RLAgent; 

class RLTrainer {
public:
    RLTrainer(int input_dim, int action_dim, int hidden);
    void collectExperience(const std::vector<float>& state,
                           const std::vector<float>& action,
                           float reward,
                           const std::vector<float>& next_state,
                           bool done);
    void trainPPO(RLAgent* agent);
    void trainSAC(RLAgent* agent);
private:
    std::vector<std::vector<float>> states_;
    std::vector<std::vector<float>> actions_;
    std::vector<float> rewards_;
    std::vector<std::vector<float>> next_states_;
    std::vector<bool> dones_;
    int capacity_;
    int position_;
    size_t batch_size_;
    std::mt19937 rng_;
};

inline RLTrainer::RLTrainer(int input_dim, int action_dim, int hidden)
    : capacity_(SAC_BUFFER_SIZE), batch_size_(PPO_BATCH_SIZE), position_(0) {
    states_.reserve(capacity_);
    actions_.reserve(capacity_);
    rewards_.reserve(capacity_);
    next_states_.reserve(capacity_);
    dones_.reserve(capacity_);
    rng_.seed(std::random_device{}());
}

inline void RLTrainer::collectExperience(const std::vector<float>& s,
                                         const std::vector<float>& a,
                                         float r,
                                         const std::vector<float>& ns,
                                         bool d) {
    if (states_.size() < capacity_) {
        states_.push_back(s);
        actions_.push_back(a);
        rewards_.push_back(r);
        next_states_.push_back(ns);
        dones_.push_back(d);
    } else {
        size_t idx = position_ % capacity_;
        states_[idx] = s;
        actions_[idx] = a;
        rewards_[idx] = r;
        next_states_[idx] = ns;
        dones_[idx] = d;
    }
    position_++;
}

inline void RLTrainer::trainPPO(RLAgent* agent) {
    if (states_.empty()) return;
    
    // Simplistic mock PPO update
    agent->epsilon *= 0.999f;
    agent->learningRate *= 0.998f;
    
    // In a real Eigen-based system:
    // MatrixXf S(states_.size(), states_[0].size());
    // ... compute advantages, update actors/critics ...
}

inline void RLTrainer::trainSAC(RLAgent* agent) {
    if (states_.empty()) return;
    // mock
}

#endif
