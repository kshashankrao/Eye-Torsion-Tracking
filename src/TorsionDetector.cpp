#include "TorsionDetector.hpp"
#include <stdexcept>
#include <utility>

TorsionDetector::TorsionDetector(std::unique_ptr<TorsionAlgorithm> algo)
    : algorithm_(std::move(algo)) {
    if (!algorithm_) {
        throw std::invalid_argument("TorsionDetector requires a non-null algorithm strategy!");
    }
}

TorsionResult TorsionDetector::process(const cv::Mat& prev_frame, 
                                       const cv::Mat& curr_frame, 
                                       bool request_diagnostics) {
    return algorithm_->calculateTorsion(prev_frame, curr_frame, request_diagnostics);
}

std::string TorsionDetector::getAlgorithmName() const {
    return algorithm_->name();
}
