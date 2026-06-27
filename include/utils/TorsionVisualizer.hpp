#pragma once
#include <opencv2/opencv.hpp>
#include "../algorithms/TorsionAlgorithm.hpp"

class TorsionVisualizer {
public:
    static TorsionDiagnostics generateDiagnostics(const TorsionIntermediates& inter,
                                                   double polar_max_radius,
                                                   int iris_inner_row,
                                                   int iris_outer_row,
                                                   const cv::Point2d& shift) {
        TorsionDiagnostics debug_info;
        debug_info.shift = shift;
        
        // 1. Clean images
        debug_info.clean_prev = inter.cleaned_prev.clone();
        debug_info.clean_curr = inter.cleaned_curr.clone();
        
        // 2. Cartesian Mask images (glint colored in red)
        cv::Mat display_mask_prev, display_mask_curr;
        inter.cartesian_mask_prev.convertTo(display_mask_prev, CV_8U, 255.0);
        cv::bitwise_not(display_mask_prev, display_mask_prev);
        inter.cartesian_mask_curr.convertTo(display_mask_curr, CV_8U, 255.0);
        cv::bitwise_not(display_mask_curr, display_mask_curr);
        
        cv::Mat cartesian_overlay_prev, cartesian_overlay_curr;
        cv::cvtColor(inter.cleaned_prev, cartesian_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(inter.cleaned_curr, cartesian_overlay_curr, cv::COLOR_GRAY2BGR);
        
        cartesian_overlay_prev.setTo(cv::Scalar(0, 0, 255), display_mask_prev);
        cartesian_overlay_curr.setTo(cv::Scalar(0, 0, 255), display_mask_curr);
        
        debug_info.mask_prev = std::move(cartesian_overlay_prev);
        debug_info.mask_curr = std::move(cartesian_overlay_curr);
        
        // 3. Polar images
        debug_info.polar_prev = inter.polar_prev.clone();
        debug_info.polar_curr = inter.polar_curr.clone();
        
        // 4. Iris cropped images
        debug_info.iris_prev = inter.iris_prev.clone();
        debug_info.iris_curr = inter.iris_curr.clone();
        
        // 5. Polar feature overlays (green dots)
        cv::Mat grad_y;
        cv::Sobel(inter.enhanced_prev, grad_y, CV_32F, 0, 1, 3);
        cv::Mat grad_mag_prev = cv::abs(grad_y);
        cv::Sobel(inter.enhanced_curr, grad_y, CV_32F, 0, 1, 3);
        cv::Mat grad_mag_curr = cv::abs(grad_y);
        
        cv::Mat valid_mask_prev_8u, valid_mask_curr_8u;
        inter.iris_mask_prev.convertTo(valid_mask_prev_8u, CV_8U, 255.0);
        inter.iris_mask_curr.convertTo(valid_mask_curr_8u, CV_8U, 255.0);
        
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
        cv::cvtColor(inter.enhanced_prev, feature_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(inter.enhanced_curr, feature_overlay_curr, cv::COLOR_GRAY2BGR);
        feature_overlay_prev.setTo(cv::Scalar(0, 255, 0), top_features_prev);
        feature_overlay_curr.setTo(cv::Scalar(0, 255, 0), top_features_curr);
        
        debug_info.grad_prev = std::move(feature_overlay_prev);
        debug_info.grad_curr = std::move(feature_overlay_curr);
        
        // 6. Map features back to Cartesian space (green overlays on raw frames)
        cv::Range iris_rows(iris_inner_row, iris_outer_row);
        cv::Mat cart_features_prev = cv::Mat::zeros(inter.prev_frame.size(), CV_8U);
        cv::Mat cart_features_curr = cv::Mat::zeros(inter.curr_frame.size(), CV_8U);
        
        cv::Mat polar_features_prev = cv::Mat::zeros(inter.polar_prev.size(), CV_8U);
        polar_features_prev(iris_rows, cv::Range::all()).setTo(255, top_features_prev);
        
        cv::Mat polar_features_curr = cv::Mat::zeros(inter.polar_curr.size(), CV_8U);
        polar_features_curr(iris_rows, cv::Range::all()).setTo(255, top_features_curr);
        
        cv::Mat raw_features_prev, raw_features_curr;
        cv::transpose(polar_features_prev, raw_features_prev);
        cv::transpose(polar_features_curr, raw_features_curr);
        
        cv::Point2f center(inter.prev_frame.cols / 2.0f, inter.prev_frame.rows / 2.0f);
        cv::warpPolar(raw_features_prev, cart_features_prev, inter.prev_frame.size(), center, polar_max_radius,
                      cv::WARP_POLAR_LINEAR | cv::INTER_NEAREST | cv::WARP_INVERSE_MAP);
        cv::warpPolar(raw_features_curr, cart_features_curr, inter.curr_frame.size(), center, polar_max_radius,
                      cv::WARP_POLAR_LINEAR | cv::INTER_NEAREST | cv::WARP_INVERSE_MAP);
                      
        cv::Mat cart_overlay_prev, cart_overlay_curr;
        cv::cvtColor(inter.prev_frame, cart_overlay_prev, cv::COLOR_GRAY2BGR);
        cv::cvtColor(inter.curr_frame, cart_overlay_curr, cv::COLOR_GRAY2BGR);
        
        cart_overlay_prev.setTo(cv::Scalar(0, 255, 0), cart_features_prev > 0);
        cart_overlay_curr.setTo(cv::Scalar(0, 255, 0), cart_features_curr > 0);
        
        debug_info.cartesian_features_prev = std::move(cart_overlay_prev);
        debug_info.cartesian_features_curr = std::move(cart_overlay_curr);
        
        return debug_info;
    }

    static void saveDiagnostics(const TorsionDiagnostics& diagnostics, const std::string& out_dir) {
        // Cartesian cleaned images (no overlays)
        cv::imwrite(out_dir + "/debug_01_clean_prev.png", diagnostics.clean_prev);
        cv::imwrite(out_dir + "/debug_01_clean_curr.png", diagnostics.clean_curr);
        
        // Cartesian mask images (red dots)
        cv::imwrite(out_dir + "/debug_01a_mask_prev.png", diagnostics.mask_prev);
        cv::imwrite(out_dir + "/debug_01a_mask_curr.png", diagnostics.mask_curr);
        
        // Polar warped images
        cv::imwrite(out_dir + "/debug_02_polar_prev.png", diagnostics.polar_prev);
        cv::imwrite(out_dir + "/debug_02_polar_curr.png", diagnostics.polar_curr);
        
        // Iris CLAHE enhanced regions (red dots indicate masked out regions in polar space)
        cv::imwrite(out_dir + "/debug_03_iris_clahe_prev.png", diagnostics.iris_prev);
        cv::imwrite(out_dir + "/debug_03_iris_clahe_curr.png", diagnostics.iris_curr);
        
        // Features used for correlation (green dots on polar space)
        cv::imwrite(out_dir + "/debug_04_features_prev.png", diagnostics.grad_prev);
        cv::imwrite(out_dir + "/debug_04_features_curr.png", diagnostics.grad_curr);
        
        // Features used for correlation mapped to Cartesian space (green dots on real eye image)
        cv::imwrite(out_dir + "/debug_04a_cartesian_features_prev.png", diagnostics.cartesian_features_prev);
        cv::imwrite(out_dir + "/debug_04a_cartesian_features_curr.png", diagnostics.cartesian_features_curr);
        
        std::cout << "\n[DEBUG] Diagnostic intermediate images saved to '" << out_dir << "/' folder:" << std::endl;
        std::cout << "  - Cleaned Cartesian: debug_01_clean_*.png" << std::endl;
        std::cout << "  - Masked Cartesian:  debug_01a_mask_*.png" << std::endl;
        std::cout << "  - Warp Polar:        debug_02_polar_*.png" << std::endl;
        std::cout << "  - Iris CLAHE:        debug_03_iris_clahe_*.png" << std::endl;
        std::cout << "  - Active Features (Polar): debug_04_features_*.png" << std::endl;
        std::cout << "  - Active Features (Cart):  debug_04a_cartesian_features_*.png" << std::endl;
        std::cout << "[DEBUG] Measured Shift: (" << diagnostics.shift.x << ", " << diagnostics.shift.y << ")" << std::endl;
    }
};
