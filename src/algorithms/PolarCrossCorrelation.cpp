#include "algorithms/PolarCrossCorrelation.hpp"
#include "utils/PerformanceTracker.hpp"
#include "utils/TorsionVisualizer.hpp"
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
    
    // Initialize cached CLAHE instances
    clahe_prev_ = cv::createCLAHE(clahe_clip_limit_, cv::Size(clahe_grid_size_, clahe_grid_size_));
    clahe_curr_ = cv::createCLAHE(clahe_clip_limit_, cv::Size(clahe_grid_size_, clahe_grid_size_));
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

void PolarCrossCorrelation::precomputePolarMaps(int width, int height) {
    cv::Point2f center(width / 2.0f, height / 2.0f);
    
    map_x_.create(cv::Size(radial_bins_, angular_bins_), CV_32FC1);
    map_y_.create(cv::Size(radial_bins_, angular_bins_), CV_32FC1);
    
    for (int y = 0; y < angular_bins_; ++y) {
        double theta = (static_cast<double>(y) / angular_bins_) * 2.0 * CV_PI;
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        
        float* mx_ptr = map_x_.ptr<float>(y);
        float* my_ptr = map_y_.ptr<float>(y);
        
        for (int x = 0; x < radial_bins_; ++x) {
            double r = (static_cast<double>(x) / radial_bins_) * polar_max_radius_;
            mx_ptr[x] = static_cast<float>(center.x + r * cos_t);
            my_ptr[x] = static_cast<float>(center.y + r * sin_t);
        }
    }
    maps_initialized_ = true;
}

cv::Mat PolarCrossCorrelation::convertToPolar(const cv::Mat& src, int interpolation) {
    if (!maps_initialized_) {
        precomputePolarMaps(src.cols, src.rows);
    }
    cv::Mat raw;
    cv::remap(src, raw, map_x_, map_y_, interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));

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
    // Lazily precompute the polar warp coordinate maps on the first call (thread-safe)
    if (!maps_initialized_) {
        precomputePolarMaps(prev_frame.cols, prev_frame.rows);
    }

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
    
    // Enhance iris texture using cached CLAHE instances and convert to CV_32F (in parallel)
    cv::Mat enhanced_prev, enhanced_curr;
    cv::Mat float_prev_32f, float_curr_32f;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            clahe_prev_->apply(iris_prev, enhanced_prev);
            enhanced_prev.convertTo(float_prev_32f, CV_32F);
        }
        #pragma omp section
        {
            clahe_curr_->apply(iris_curr, enhanced_curr);
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
        TorsionIntermediates inter {
            prev_frame,
            curr_frame,
            cleaned_prev,
            cleaned_curr,
            cartesian_mask_prev,
            cartesian_mask_curr,
            polar_prev,
            polar_curr,
            iris_prev,
            iris_curr,
            enhanced_prev,
            enhanced_curr,
            iris_mask_prev,
            iris_mask_curr
        };
        result.diagnostics = TorsionVisualizer::generateDiagnostics(
            inter,
            polar_max_radius_,
            iris_inner_row_,
            iris_outer_row_,
            cv::Point2d(0, shift_y)
        );
    }
    
    return result;
}
