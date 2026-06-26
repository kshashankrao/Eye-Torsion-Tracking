#include "algorithms/PolarCrossCorrelation.hpp"
#include <opencv2/photo.hpp>
#include <iostream>
#include <utility>

PolarCrossCorrelation::PolarCrossCorrelation(int radial_bins, int angular_bins)
    : radial_bins_(radial_bins), angular_bins_(angular_bins) {}

cv::Mat PolarCrossCorrelation::removeGlints(const cv::Mat& src) {
    // 1. Create a mask of the bright glints (saturated pixels)
    cv::Mat mask;
    cv::threshold(src, mask, 240, 255, cv::THRESH_BINARY);
    
    // 2. Dilate the mask to cover the transitional edges of the glints
    cv::Mat dilated_mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(mask, dilated_mask, kernel);
    
    // 3. Inpaint the glint regions using Telea's method
    cv::Mat dst;
    cv::inpaint(src, dilated_mask, dst, 3.0, cv::INPAINT_TELEA);
    return dst;
}

cv::Mat PolarCrossCorrelation::convertToPolar(const cv::Mat& src) {
    cv::Mat dst;
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    
    // We target the iris region. The crop is 160x160.
    // Half-width is 80.
    double maxRadius = 75.0; 
    
    // cv::warpPolar maps radius to the width (cols) and theta to the height (rows)
    cv::warpPolar(src, dst, cv::Size(radial_bins_, angular_bins_), center, maxRadius, 
                  cv::WARP_POLAR_LINEAR | cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS);
    return dst;
}

TorsionResult PolarCrossCorrelation::calculateTorsion(const cv::Mat& prev_frame, 
                                                      const cv::Mat& curr_frame, 
                                                      bool request_diagnostics) {
    // 1. Preprocess: remove stationary glints in Cartesian space
    cv::Mat cleaned_prev = removeGlints(prev_frame);
    cv::Mat cleaned_curr = removeGlints(curr_frame);

    // 2. Convert Cartesian images to Polar coordinates
    cv::Mat polar_prev = convertToPolar(cleaned_prev);
    cv::Mat polar_curr = convertToPolar(cleaned_curr);
    
    // ----------------------------------------------------
    // IRIS ANNULUS EXTRACTION & CONTRAST ENHANCEMENT
    // ----------------------------------------------------
    // Note: Since columns (X-axis) represent RADIUS in OpenCV's warpPolar,
    // we slice along columns (X-axis) instead of rows (Y-axis) to isolate the iris annulus.
    // Out of radial_bins_ (80 bins), columns 35 to 70 represent the iris.
    cv::Range iris_cols(35, 70);
    cv::Mat iris_prev = polar_prev(cv::Range::all(), iris_cols);
    cv::Mat iris_curr = polar_curr(cv::Range::all(), iris_cols);
    
    // Enhance iris texture using CLAHE (applied to columns representing the iris)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat enhanced_prev, enhanced_curr;
    clahe->apply(iris_prev, enhanced_prev);
    clahe->apply(iris_curr, enhanced_curr);
    
    // 3. Convert to float for Fourier Transform inside phaseCorrelate (yielding the best accuracy)
    cv::Mat float_prev_32f, float_curr_32f;
    enhanced_prev.convertTo(float_prev_32f, CV_32F);
    enhanced_curr.convertTo(float_curr_32f, CV_32F);
    
    // 4. Compute Phase Correlation on the CLAHE-enhanced images
    double peak_response = 0.0;
    cv::Point2d shift = cv::phaseCorrelate(float_prev_32f, float_curr_32f, cv::noArray(), &peak_response);
    
    // 5. Highlight vertical features (angular shifts) using a vertical Sobel filter (for visualization/diagnostics)
    cv::Mat grad_prev_32f, grad_curr_32f;
    cv::Sobel(enhanced_prev, grad_prev_32f, CV_32F, 0, 1, 3);
    cv::Sobel(enhanced_curr, grad_curr_32f, CV_32F, 0, 1, 3);
    
    // Translate shift along vertical axis (Y-axis, which is THETA!) to degrees
    // The polar image height (angular_bins_ = 360) maps to 360 degrees.
    // Note: We multiply by -1.0 to correct the sign flip mismatch between Cartesian and Polar rotations.
    double angle_deg = -1.0 * (shift.y / angular_bins_) * 360.0;
    
    // Normalize to [-180, 180]
    while (angle_deg > 180.0) angle_deg -= 360.0;
    while (angle_deg < -180.0) angle_deg += 360.0;
    
    // 6. Build TorsionResult
    TorsionResult result;
    result.angle = angle_deg;
    result.confidence = peak_response;
    result.success = (peak_response > 0.01);
    
    if (request_diagnostics) {
        TorsionDiagnostics debug_info;
        debug_info.clean_prev = cleaned_prev;
        debug_info.clean_curr = cleaned_curr;
        debug_info.polar_prev = polar_prev;
        debug_info.polar_curr = polar_curr;
        debug_info.iris_prev = enhanced_prev;
        debug_info.iris_curr = enhanced_curr;
        debug_info.grad_prev = grad_prev_32f;
        debug_info.grad_curr = grad_curr_32f;
        debug_info.shift = shift;
        
        result.diagnostics = std::move(debug_info);
    }
    
    return result;
}
