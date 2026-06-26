#pragma once

#include "algorithms/TorsionAlgorithm.hpp"

class PolarCrossCorrelation : public TorsionAlgorithm {
public:
    /**
     * @brief Construct the PolarCrossCorrelation algorithm.
     * @param radial_bins The number of radial bins (width of polar image).
     * @param angular_bins The number of angular bins (height of polar image, maps to 360 degrees).
     */
    explicit PolarCrossCorrelation(int radial_bins = 80, int angular_bins = 360);

    TorsionResult calculateTorsion(const cv::Mat& prev_frame, 
                                   const cv::Mat& curr_frame, 
                                   bool request_diagnostics = kDefaultRequestDiagnostics) override;

    std::string name() const override { return "PolarCrossCorrelation"; }

private:
    int radial_bins_;
    int angular_bins_;

    cv::Mat convertToPolar(const cv::Mat& src);
    cv::Mat removeGlints(const cv::Mat& src);
};
