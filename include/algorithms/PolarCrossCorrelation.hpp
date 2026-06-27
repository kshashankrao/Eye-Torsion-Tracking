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

    // Hyperparameters
    int glint_threshold_;
    int glint_kernel_size_;
    double glint_inpaint_radius_;
    double polar_max_radius_;
    int iris_inner_row_;
    int iris_outer_row_;
    double clahe_clip_limit_;
    int clahe_grid_size_;
    int max_search_shift_deg_;

    cv::Mat convertToPolar(const cv::Mat& src, int interpolation = cv::INTER_LINEAR);
    std::pair<cv::Mat, cv::Mat> removeGlints(const cv::Mat& src);
    double calculateMaskedNCC(const cv::Mat& prev, const cv::Mat& curr, const cv::Mat& mask_prev, const cv::Mat& mask_curr, int max_shift, double& best_peak);

    // Cache pre-computed Polar coordinates LUT mapping
    cv::Mat map_x_, map_y_;
    bool maps_initialized_ = false;
    void precomputePolarMaps(int width, int height);

    // Cache CLAHE instances to avoid heap allocation in main loops
    cv::Ptr<cv::CLAHE> clahe_prev_;
    cv::Ptr<cv::CLAHE> clahe_curr_;
};
