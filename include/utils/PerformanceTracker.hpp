#pragma once
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

class PerformanceTracker {
public:
    void start(const std::string& name = "total") {
        active_starts_[name] = std::chrono::high_resolution_clock::now();
    }

    double stop(const std::string& name = "total") {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto it = active_starts_.find(name);
        if (it == active_starts_.end()) return 0.0;
        
        double elapsed = std::chrono::duration<double, std::milli>(end_time - it->second).count();
        runtimes_[name].push_back(elapsed);
        return elapsed;
    }

    double getMean(const std::string& name = "total") const {
        auto it = runtimes_.find(name);
        if (it == runtimes_.end() || it->second.empty()) return 0.0;
        double sum = std::accumulate(it->second.begin(), it->second.end(), 0.0);
        return sum / it->second.size();
    }

    double getMin(const std::string& name = "total") const {
        auto it = runtimes_.find(name);
        if (it == runtimes_.end() || it->second.empty()) return 0.0;
        return *std::min_element(it->second.begin(), it->second.end());
    }

    double getMax(const std::string& name = "total") const {
        auto it = runtimes_.find(name);
        if (it == runtimes_.end() || it->second.empty()) return 0.0;
        return *std::max_element(it->second.begin(), it->second.end());
    }

    double getStdDev(const std::string& name = "total") const {
        auto it = runtimes_.find(name);
        if (it == runtimes_.end() || it->second.size() <= 1) return 0.0;
        double mean = getMean(name);
        double accum = 0.0;
        for (double val : it->second) {
            accum += (val - mean) * (val - mean);
        }
        return std::sqrt(accum / (it->second.size() - 1));
    }

    void reset() {
        active_starts_.clear();
        runtimes_.clear();
    }

    void printReport() const {
        std::cout << "\n=======================================================" << std::endl;
        std::cout << "                STAGE PROFILING REPORT                 " << std::endl;
        std::cout << "=======================================================" << std::endl;
        std::cout << std::left << std::setw(25) << "Stage Name" 
                  << std::right << std::setw(10) << "Mean (ms)" 
                  << std::setw(10) << "Min (ms)" 
                  << std::setw(10) << "Max (ms)" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        for (const auto& [name, runs] : runtimes_) {
            if (runs.empty()) continue;
            double sum = std::accumulate(runs.begin(), runs.end(), 0.0);
            double mean = sum / runs.size();
            double min_val = *std::min_element(runs.begin(), runs.end());
            double max_val = *std::max_element(runs.begin(), runs.end());
            std::cout << std::left << std::setw(25) << name 
                      << std::right << std::setw(10) << std::fixed << std::setprecision(3) << mean 
                      << std::setw(10) << min_val 
                      << std::setw(10) << max_val << std::endl;
        }
        std::cout << "=======================================================" << std::endl;
    }

private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> active_starts_;
    std::map<std::string, std::vector<double>> runtimes_;
};
