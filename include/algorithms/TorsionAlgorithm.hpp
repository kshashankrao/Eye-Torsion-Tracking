#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

struct TorsionDiagnostics {
    cv::Mat clean_prev;
    cv::Mat clean_curr;
    cv::Mat mask_prev;
    cv::Mat mask_curr;
    cv::Mat polar_prev;
    cv::Mat polar_curr;
    cv::Mat iris_prev;
    cv::Mat iris_curr;
    cv::Mat grad_prev;
    cv::Mat grad_curr;
    cv::Mat cartesian_features_prev;
    cv::Mat cartesian_features_curr;
    cv::Point2d shift;
};

struct TorsionResult {
    double angle;
    double confidence;
    bool success;
    std::optional<TorsionDiagnostics> diagnostics;
};

#ifndef DIAGNOSTICS_DEFAULT
#define DIAGNOSTICS_DEFAULT false
#endif

constexpr bool kDefaultRequestDiagnostics = DIAGNOSTICS_DEFAULT;

class TorsionAlgorithm {
public:
    virtual ~TorsionAlgorithm() = default;

    /**
     * @brief Get the class name of the algorithm.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Calculates the eye torsion roll angle between two frames.
     * @param prev_frame Grayscale image at time t-1.
     * @param curr_frame Grayscale image at time t.
     * @param request_diagnostics If true, returns intermediate matrices in the diagnostics field.
     * @return TorsionResult structure containing the estimated angle and diagnostics.
     */
    virtual TorsionResult calculateTorsion(const cv::Mat& prev_frame, 
                                           const cv::Mat& curr_frame, 
                                           bool request_diagnostics = kDefaultRequestDiagnostics) = 0;
};
