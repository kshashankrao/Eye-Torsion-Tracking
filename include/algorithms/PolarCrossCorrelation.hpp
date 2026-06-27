#pragma once

#include "algorithms/TorsionAlgorithm.hpp"

class PolarCrossCorrelation : public TorsionAlgorithm {
public:
    /**
     * @brief Construct the PolarCrossCorrelation algorithm.
     * @param radial_bins The number of radial bins (width of polar image).
     * @param angular_bins The number of angular bins (height of polar image, maps to 360 degrees).
     */
    explicit PolarCrossCorrelation(int radial_bins = 80, int angular_bins = 360, bool use_fft = false);

    TorsionResult calculateTorsion(const cv::Mat& prev_frame, 
                                   const cv::Mat& curr_frame, 
                                   bool request_diagnostics = kDefaultRequestDiagnostics) override;

    std::string name() const override { 
        return use_fft_ ? "PolarCrossCorrelation_FFT" : "PolarCrossCorrelation_Masked"; 
    }

protected:
    int radial_bins_;
    int angular_bins_;
    bool use_fft_;

    cv::Mat convertToPolar(const cv::Mat& src, int interpolation = cv::INTER_LINEAR);
    std::pair<cv::Mat, cv::Mat> removeGlints(const cv::Mat& src);
    double calculateMaskedNCC(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& mask_prev, const cv::Mat& mask_curr, int max_shift, double& best_peak);
};
