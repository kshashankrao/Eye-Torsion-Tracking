#include "algorithms/PolarCrossCorrelation.hpp"
#include "utils/PerformanceTracker.hpp"
#include <opencv2/photo.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <utility>

PerformanceTracker g_algo_tracker;

PolarCrossCorrelation::PolarCrossCorrelation(int radial_bins, int angular_bins, bool use_fft)
    : radial_bins_(radial_bins),
      angular_bins_(angular_bins),
      use_fft_(use_fft),
      glint_threshold_(140),
      glint_kernel_size_(11),
      glint_inpaint_radius_(5.0),
      polar_max_radius_(75.0),
      iris_inner_row_(35),
      iris_outer_row_(70),
      clahe_clip_limit_(3.0),
      clahe_grid_size_(8),
      max_search_shift_deg_(45) {
    try {
        std::ifstream file("config/config.json");
        if (file.is_open()) {
            nlohmann::json config;
            file >> config;
            if (config.contains("algorithm")) {
                auto alg = config["algorithm"];
                if (alg.contains("radial_bins")) radial_bins_ = alg["radial_bins"].get<int>();
                if (alg.contains("angular_bins")) angular_bins_ = alg["angular_bins"].get<int>();
                if (alg.contains("use_fft")) use_fft_ = alg["use_fft"].get<bool>();
                
                if (alg.contains("glint_threshold")) glint_threshold_ = alg["glint_threshold"].get<int>();
                if (alg.contains("glint_kernel_size")) glint_kernel_size_ = alg["glint_kernel_size"].get<int>();
                if (alg.contains("glint_inpaint_radius")) glint_inpaint_radius_ = alg["glint_inpaint_radius"].get<double>();
                if (alg.contains("polar_max_radius")) polar_max_radius_ = alg["polar_max_radius"].get<double>();
                if (alg.contains("iris_inner_row")) iris_inner_row_ = alg["iris_inner_row"].get<int>();
                if (alg.contains("iris_outer_row")) iris_outer_row_ = alg["iris_outer_row"].get<int>();
                if (alg.contains("clahe_clip_limit")) clahe_clip_limit_ = alg["clahe_clip_limit"].get<double>();
                if (alg.contains("clahe_grid_size")) clahe_grid_size_ = alg["clahe_grid_size"].get<int>();
                if (alg.contains("max_search_shift_deg")) max_search_shift_deg_ = alg["max_search_shift_deg"].get<int>();
            }
        }
    } catch (...) {
        // Fallback to defaults
    }
}

std::pair<cv::Mat, cv::Mat> PolarCrossCorrelation::removeGlints(const cv::Mat& src) {
    // 1. Create a mask of the bright glints (saturated pixels)
    cv::Mat mask;
    cv::threshold(src, mask, glint_threshold_, 255, cv::THRESH_BINARY);
    
    // 2. Dilate the mask to cover the transitional edges and halos of the glints
    cv::Mat dilated_mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(glint_kernel_size_, glint_kernel_size_));
    cv::dilate(mask, dilated_mask, kernel);
    
    // Create valid mask (inverted): valid pixels are 1, glints are 0
    cv::Mat valid_mask;
    cv::bitwise_not(dilated_mask, valid_mask);
    valid_mask.setTo(1, valid_mask > 0);
    
    // 3. Fast fill: set dilated mask regions to the average intensity of valid pixels
    // This prevents CLAHE boundary artifacts while running in 0.2ms
    cv::Scalar mean_val = cv::mean(src, valid_mask);
    cv::Mat dst = src.clone();
    dst.setTo(mean_val, dilated_mask);
    
    return {dst, valid_mask};
}

cv::Mat PolarCrossCorrelation::convertToPolar(const cv::Mat& src, int interpolation) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);

    // We target the iris region. The crop is 160x160.
    // Half-width is 80.
    double maxRadius = polar_max_radius_;

    // OpenCV's warpPolar maps: rows → angle (phi), cols → radius (rho).
    // To get the INTENDED layout (rows = radius, cols = angle) we must:
    //   1. Call warpPolar with cv::Size(radial_bins_, angular_bins_) so that the
    //      intermediate image is (angular_bins_ rows) × (radial_bins_ cols).
    //   2. Transpose the result to produce (radial_bins_ rows) × (angular_bins_ cols).
    // After the transpose: rows = radial_bins_ (radius), cols = angular_bins_ (angle).
    // This lets the NCC shift horizontally (in cols = angle) to detect torsion.
    cv::Mat raw;
    cv::warpPolar(src, raw, cv::Size(radial_bins_, angular_bins_), center, maxRadius,
                  cv::WARP_POLAR_LINEAR | interpolation | cv::WARP_FILL_OUTLIERS);

    cv::Mat dst;
    cv::transpose(raw, dst);
    return dst;
}


double PolarCrossCorrelation::calculateMaskedNCC(const cv::Mat& prev,
                                                const cv::Mat& curr,
                                                const cv::Mat& mask_prev,
                                                const cv::Mat& mask_curr,
                                                int max_shift, double& best_peak) {
    // Dimensions
    const int height = prev.rows;
    const int width = prev.cols;

    std::vector<double> ncc_scores(2 * max_shift + 1, 0.0);

    // For each possible column shift (dx) compute NCC in a parallel loop.
    #pragma omp parallel for
    for (int dx = -max_shift; dx <= max_shift; ++dx) {
        // Accumulators for the current shift
        double sum_p = 0.0, sum_c = 0.0;
        double sum_pp = 0.0, sum_cc = 0.0, sum_pc = 0.0;
        int valid_count = 0;

        // Iterate over all rows and columns
        for (int y = 0; y < height; ++y) {
            const float* p_row = prev.ptr<float>(y);
            const float* c_row = curr.ptr<float>(y);
            const uchar* mp_row = mask_prev.ptr<uchar>(y);
            const uchar* mc_row = mask_curr.ptr<uchar>(y);

            // Compute wrapped column index once per x to avoid repeated modulo.
            for (int x = 0; x < width; ++x) {
                int xc = x + dx;
                // manual wrap (faster than % when dx is small)
                if (xc >= width) xc -= width;
                else if (xc < 0) xc += width;

                if (mp_row[x] > 0 && mc_row[xc] > 0) {
                    const double p_val = static_cast<double>(p_row[x]);
                    const double c_val = static_cast<double>(c_row[xc]);
                    sum_p += p_val;
                    sum_c += c_val;
                    sum_pp += p_val * p_val;
                    sum_cc += c_val * c_val;
                    sum_pc += p_val * c_val;
                    ++valid_count;
                }
            }
        }

        // No overlapping valid pixels – set a sentinel low NCC.
        if (valid_count == 0) {
            ncc_scores[dx + max_shift] = -1.0;
            continue;
        }

        // Compute means
        const double mean_p = sum_p / valid_count;
        const double mean_c = sum_c / valid_count;

        // Compute NCC using the accumulated sums (single‑pass formula)
        const double numerator   = sum_pc - sum_p * mean_c; // = sum((p-mean_p)*(c-mean_c))
        const double denom_p_sq  = sum_pp - sum_p * mean_p; // = sum((p-mean_p)^2)
        const double denom_c_sq  = sum_cc - sum_c * mean_c; // = sum((c-mean_c)^2)
        const double denominator = std::sqrt(denom_p_sq * denom_c_sq);
        const double ncc = (denominator == 0.0) ? 0.0 : numerator / denominator;

        ncc_scores[dx + max_shift] = ncc;
    }

    // Find the best shift sequentially
    double best_ncc = -1.0;
    int best_dx = 0;
    for (int dx = -max_shift; dx <= max_shift; ++dx) {
        double ncc = ncc_scores[dx + max_shift];
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_dx = dx;
        }
    }

    best_peak = best_ncc;

    // Sub‑pixel parabola refinement (if the peak is not at the border)
    if (best_dx > -max_shift && best_dx < max_shift) {
        const double y1 = ncc_scores[best_dx - 1 + max_shift];
        const double y2 = ncc_scores[best_dx + max_shift];
        const double y3 = ncc_scores[best_dx + 1 + max_shift];
        const double denom = 2.0 * (y1 - 2.0 * y2 + y3);
        if (denom != 0.0) {
            const double sub = (y1 - y3) / denom;
            // Return raw positive shift; calculateTorsion applies its own -1.0 sign flip
            return static_cast<double>(best_dx) + sub;
        }
    }

    // Return the raw integer shift (calculateTorsion applies its own -1.0 sign flip)
    return static_cast<double>(best_dx);
}


TorsionResult PolarCrossCorrelation::calculateTorsion(const cv::Mat& prev_frame, 
                                                      const cv::Mat& curr_frame, 
                                                      bool request_diagnostics) {
    // 1. Preprocess: remove stationary glints in Cartesian space (in parallel)
    cv::Mat cleaned_prev, cleaned_curr, cartesian_mask_prev, cartesian_mask_curr;
    g_algo_tracker.start("1. Glint Removal");
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto [p, m] = removeGlints(prev_frame);
            cleaned_prev = p;
            cartesian_mask_prev = m;
        }
        #pragma omp section
        {
            auto [c, m] = removeGlints(curr_frame);
            cleaned_curr = c;
            cartesian_mask_curr = m;
        }
    }
    g_algo_tracker.stop("1. Glint Removal");

    // 2. Convert Cartesian images to Polar coordinates (all 4 warps in parallel)
    cv::Mat polar_prev, polar_curr, polar_mask_prev, polar_mask_curr;
    g_algo_tracker.start("2. Polar Warp");
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            polar_prev = convertToPolar(cleaned_prev);
        }
        #pragma omp section
        {
            polar_curr = convertToPolar(cleaned_curr);
        }
        #pragma omp section
        {
            polar_mask_prev = convertToPolar(cartesian_mask_prev, cv::INTER_NEAREST);
        }
        #pragma omp section
        {
            polar_mask_curr = convertToPolar(cartesian_mask_curr, cv::INTER_NEAREST);
        }
    }
    g_algo_tracker.stop("2. Polar Warp");
    
    // ----------------------------------------------------
    // IRIS ANNULUS EXTRACTION & CONTRAST ENHANCEMENT
    // ----------------------------------------------------
    g_algo_tracker.start("3. Iris Crop & CLAHE");
    // Select the iris annulus: radial rows 35–70 out of 80 radial bins.
    // Rows = radius, Cols = angle (warpPolar output: rows=rho, cols=phi).
    // This gives a 35×1440 sub-image so the NCC can shift in the full angular dimension.
    cv::Range iris_rows(iris_inner_row_, iris_outer_row_);
    cv::Mat iris_prev = polar_prev(iris_rows, cv::Range::all());
    cv::Mat iris_curr = polar_curr(iris_rows, cv::Range::all());

    cv::Mat iris_mask_prev = polar_mask_prev(iris_rows, cv::Range::all());
    cv::Mat iris_mask_curr = polar_mask_curr(iris_rows, cv::Range::all());
    
    // Enhance iris texture using CLAHE and convert to CV_32F (in parallel)
    cv::Mat enhanced_prev, enhanced_curr;
    cv::Mat float_prev_32f, float_curr_32f;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_clip_limit_, cv::Size(clahe_grid_size_, clahe_grid_size_));
            clahe->apply(iris_prev, enhanced_prev);
            enhanced_prev.convertTo(float_prev_32f, CV_32F);
        }
        #pragma omp section
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_clip_limit_, cv::Size(clahe_grid_size_, clahe_grid_size_));
            clahe->apply(iris_curr, enhanced_curr);
            enhanced_curr.convertTo(float_curr_32f, CV_32F);
        }
    }
    g_algo_tracker.stop("3. Iris Crop & CLAHE");
    
    // 4. Compute tracking
    g_algo_tracker.start("4. Correlation Matching");
    double peak_response = 0.0;
    double shift_y = 0.0;
    
    if (use_fft_) {
        // Use standard OpenCV phase correlation (FFT) on the inpainted image
        cv::Point2d shift = cv::phaseCorrelate(float_prev_32f, float_curr_32f, cv::noArray(), &peak_response);
        shift_y = shift.y;
    } else {
        // Use custom Masked Spatial NCC
        int max_shift = static_cast<int>(std::round(max_search_shift_deg_ * (angular_bins_ / 360.0)));
        shift_y = calculateMaskedNCC(float_prev_32f, float_curr_32f, iris_mask_prev, iris_mask_curr, max_shift, peak_response);
    }
    g_algo_tracker.stop("4. Correlation Matching");
    
    // Translate shift along vertical axis to degrees
    // Note: We multiply by -1.0 to correct the sign flip mismatch between Cartesian and Polar rotations.
    double angle_deg = -1.0 * (shift_y / angular_bins_) * 360.0;
    
    // Normalize to [-180, 180]
    while (angle_deg > 180.0) angle_deg -= 360.0;
    while (angle_deg < -180.0) angle_deg += 360.0;
    
    // 6. Build TorsionResult
    TorsionResult result;
    result.angle = angle_deg;
    result.confidence = peak_response;
    result.success = (peak_response > 0.01);
    
    if (request_diagnostics) {
        // Prepare Cartesian masks for BGR overlay
        cv::Mat display_mask_prev, display_mask_curr;
        cartesian_mask_prev.convertTo(display_mask_prev, CV_8U, 255.0); // 1 to 255 for glint-free
        cv::bitwise_not(display_mask_prev, display_mask_prev); // 255 for glint
        
        cartesian_mask_curr.convertTo(display_mask_curr, CV_8U, 255.0);
        cv::bitwise_not(display_mask_curr, display_mask_curr);
        
        // 3. Create BGR overlays for the Cartesian real eye crops (MASK)
        cv::Mat cartesian_overlay_prev, cartesian_overlay_curr;
        cv::cvtColor(cleaned_prev, cartesian_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cleaned_curr, cartesian_overlay_curr, cv::COLOR_GRAY2BGR);
        
        // Color the glint masks red in Cartesian coordinates
        cartesian_overlay_prev.setTo(cv::Scalar(0, 0, 255), display_mask_prev);
        cartesian_overlay_curr.setTo(cv::Scalar(0, 0, 255), display_mask_curr);
        
        TorsionDiagnostics debug_info;
        // Clean images (no overlay)
        debug_info.clean_prev = cleaned_prev.clone();
        debug_info.clean_curr = cleaned_curr.clone();
        
        // Mask images (red overlay)
        debug_info.mask_prev = std::move(cartesian_overlay_prev);
        debug_info.mask_curr = std::move(cartesian_overlay_curr);
        
        debug_info.polar_prev = polar_prev;
        debug_info.polar_curr = polar_curr;
        
        // To visualize features used for correlation (green dots on polar)
        cv::Mat grad_y;
        cv::Sobel(enhanced_prev, grad_y, CV_32F, 0, 1, 3);
        cv::Mat grad_mag_prev = cv::abs(grad_y);
        cv::Sobel(enhanced_curr, grad_y, CV_32F, 0, 1, 3);
        cv::Mat grad_mag_curr = cv::abs(grad_y);
        
        // Only keep gradients where mask is valid
        cv::Mat valid_mask_prev_8u, valid_mask_curr_8u;
        iris_mask_prev.convertTo(valid_mask_prev_8u, CV_8U, 255.0);
        iris_mask_curr.convertTo(valid_mask_curr_8u, CV_8U, 255.0);
        
        cv::Mat masked_grad_prev = cv::Mat::zeros(grad_mag_prev.size(), CV_32F);
        cv::Mat masked_grad_curr = cv::Mat::zeros(grad_mag_curr.size(), CV_32F);
        grad_mag_prev.copyTo(masked_grad_prev, valid_mask_prev_8u);
        grad_mag_curr.copyTo(masked_grad_curr, valid_mask_curr_8u);
        
        double max_prev = 0, max_curr = 0;
        cv::minMaxLoc(masked_grad_prev, nullptr, &max_prev);
        cv::minMaxLoc(masked_grad_curr, nullptr, &max_curr);
        
        cv::Mat top_features_prev, top_features_curr;
        cv::threshold(masked_grad_prev, top_features_prev, 0.40 * max_prev, 255, cv::THRESH_BINARY);
        cv::threshold(masked_grad_curr, top_features_curr, 0.40 * max_curr, 255, cv::THRESH_BINARY);
        top_features_prev.convertTo(top_features_prev, CV_8U);
        top_features_curr.convertTo(top_features_curr, CV_8U);
        
        cv::Mat feature_overlay_prev, feature_overlay_curr;
        cv::cvtColor(enhanced_prev, feature_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(enhanced_curr, feature_overlay_curr, cv::COLOR_GRAY2BGR);
        feature_overlay_prev.setTo(cv::Scalar(0, 255, 0), top_features_prev); // Green features
        feature_overlay_curr.setTo(cv::Scalar(0, 255, 0), top_features_curr);
        
        // Store features in grad_prev/curr
        debug_info.grad_prev = std::move(feature_overlay_prev);
        debug_info.grad_curr = std::move(feature_overlay_curr);
        
        // Warp features back to Cartesian
        cv::Mat full_feature_mask_prev = cv::Mat::zeros(polar_prev.size(), CV_8U);
        cv::Mat full_feature_mask_curr = cv::Mat::zeros(polar_curr.size(), CV_8U);
        top_features_prev.copyTo(full_feature_mask_prev(cv::Range(iris_inner_row_, iris_outer_row_), cv::Range::all()));
        top_features_curr.copyTo(full_feature_mask_curr(cv::Range(iris_inner_row_, iris_outer_row_), cv::Range::all()));
        
        cv::Mat cartesian_feature_mask_prev, cartesian_feature_mask_curr;
        cv::Point2f center(cleaned_prev.cols / 2.0f, cleaned_prev.rows / 2.0f);
        double maxRadius = polar_max_radius_;
        
        cv::warpPolar(full_feature_mask_prev, cartesian_feature_mask_prev, cleaned_prev.size(), center, maxRadius,
                      cv::WARP_INVERSE_MAP | cv::INTER_NEAREST | cv::WARP_FILL_OUTLIERS);
        cv::warpPolar(full_feature_mask_curr, cartesian_feature_mask_curr, cleaned_curr.size(), center, maxRadius,
                      cv::WARP_INVERSE_MAP | cv::INTER_NEAREST | cv::WARP_FILL_OUTLIERS);
                      
        cv::Mat cartesian_feature_overlay_prev, cartesian_feature_overlay_curr;
        cv::cvtColor(cleaned_prev, cartesian_feature_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(cleaned_curr, cartesian_feature_overlay_curr, cv::COLOR_GRAY2BGR);
        
        cartesian_feature_overlay_prev.setTo(cv::Scalar(0, 255, 0), cartesian_feature_mask_prev);
        cartesian_feature_overlay_curr.setTo(cv::Scalar(0, 255, 0), cartesian_feature_mask_curr);
        
        debug_info.cartesian_features_prev = std::move(cartesian_feature_overlay_prev);
        debug_info.cartesian_features_curr = std::move(cartesian_feature_overlay_curr);
        
        // To visualize polar mask
        cv::Mat overlay_prev, overlay_curr;
        cv::cvtColor(enhanced_prev, overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(enhanced_curr, overlay_curr, cv::COLOR_GRAY2BGR);
        cv::Mat disp_pmask_prev, disp_pmask_curr;
        iris_mask_prev.convertTo(disp_pmask_prev, CV_8U, 255.0);
        cv::bitwise_not(disp_pmask_prev, disp_pmask_prev);
        iris_mask_curr.convertTo(disp_pmask_curr, CV_8U, 255.0);
        cv::bitwise_not(disp_pmask_curr, disp_pmask_curr);
        overlay_prev.setTo(cv::Scalar(0, 0, 255), disp_pmask_prev);
        overlay_curr.setTo(cv::Scalar(0, 0, 255), disp_pmask_curr);
        
        debug_info.iris_prev = std::move(overlay_prev);
        debug_info.iris_curr = std::move(overlay_curr);
        debug_info.shift = cv::Point2d(0, shift_y);
        
        result.diagnostics = std::move(debug_info);
    }
    
    return result;
}
