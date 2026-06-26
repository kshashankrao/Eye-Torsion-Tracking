#pragma once

#include "algorithms/TorsionAlgorithm.hpp"
#include <memory>

class TorsionDetector {
public:
    /**
     * @brief Construct a TorsionDetector with a specific algorithm strategy.
     * @param algo Smart pointer taking ownership of the chosen strategy.
     */
    explicit TorsionDetector(std::unique_ptr<TorsionAlgorithm> algo);

    /**
     * @brief Process a pair of frames and calculate the torsion.
     * @param prev_frame Grayscale image at t-1.
     * @param curr_frame Grayscale image at t.
     * @param request_diagnostics If true, returns intermediate matrices in the diagnostics field.
     * @return TorsionResult structure containing results and optional diagnostics.
     */
    TorsionResult process(const cv::Mat& prev_frame, 
                          const cv::Mat& curr_frame, 
                          bool request_diagnostics = kDefaultRequestDiagnostics);

    /**
     * @brief Get the name of the wrapped algorithm strategy.
     */
    std::string getAlgorithmName() const;

private:
    std::unique_ptr<TorsionAlgorithm> algorithm_;
};
