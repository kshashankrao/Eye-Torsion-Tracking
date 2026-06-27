#pragma once
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class PerformanceTracker {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        runtimes_.push_back(elapsed);
        return elapsed;
    }

    double getMean() const {
        if (runtimes_.empty()) return 0.0;
        double sum = std::accumulate(runtimes_.begin(), runtimes_.end(), 0.0);
        return sum / runtimes_.size();
    }

    double getMin() const {
        if (runtimes_.empty()) return 0.0;
        return *std::min_element(runtimes_.begin(), runtimes_.end());
    }

    double getMax() const {
        if (runtimes_.empty()) return 0.0;
        return *std::max_element(runtimes_.begin(), runtimes_.end());
    }

    double getStdDev() const {
        if (runtimes_.size() <= 1) return 0.0;
        double mean = getMean();
        double accum = 0.0;
        for (double val : runtimes_) {
            accum += (val - mean) * (val - mean);
        }
        return std::sqrt(accum / (runtimes_.size() - 1));
    }

    void reset() {
        runtimes_.clear();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::vector<double> runtimes_;
};
