#include "DataLoader.hpp"
#include "TorsionDetector.hpp"
#include "algorithms/TorsionAlgorithm.hpp"
#include "algorithms/PolarCrossCorrelation.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>

std::string getCurrentDateTimeString() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    // Format: YYYYMMDD_HHMMSS
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

int main() {
    try {
        std::string config_path = "config/config.json";
        std::cout << "Starting Eye Torsion Tracker C++ Pipeline..." << std::endl;
        
        // 1. Initialize our lazy DataLoader iterator
        DataLoader loader(config_path);
        
        // 2. Initialize TorsionDetector with Polar Cross-Correlation Strategy
        auto polar_algo = std::make_unique<PolarCrossCorrelation>();
        TorsionDetector detector(std::move(polar_algo));
        
        // 3. Generate dynamic output folder name: output/YYYYMMDD_HHMMSS_AlgorithmName
        std::string timestamp = getCurrentDateTimeString();
        std::string algo_name = detector.getAlgorithmName();
        std::string out_dir = "output/" + timestamp + "_" + algo_name;
        
        std::filesystem::create_directories(out_dir);
        std::string out_csv_path = out_dir + "/algorithm_results.csv";
        std::ofstream out_file(out_csv_path);
        
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open output file: " << out_csv_path << std::endl;
            return 1;
        }
        
        // Write header
        out_file << "sequence,img_prev,img_curr,gt_angle,algo_angle,confidence,success\n";
        
        // 4. Iterate through samples lazily
        size_t processed_count = 0;
        bool debug_saved = false;
        
        while (loader.hasNext()) {
            TorsionSample sample = loader.next();
            
            // We pass request_diagnostics = true so we can inspect intermediate steps in debug mode!
            auto [predicted_angle, confidence, success, diagnostics] = detector.process(sample.img_prev, sample.img_curr, true);
            
            // Save one pair of diagnostic images inside the dynamic output directory
            if (diagnostics && !debug_saved) {
                // Cartesian cleaned images
                cv::imwrite(out_dir + "/debug_01_clean_prev.png", diagnostics->clean_prev);
                cv::imwrite(out_dir + "/debug_01_clean_curr.png", diagnostics->clean_curr);
                
                // Polar warped images
                cv::imwrite(out_dir + "/debug_02_polar_prev.png", diagnostics->polar_prev);
                cv::imwrite(out_dir + "/debug_02_polar_curr.png", diagnostics->polar_curr);
                
                // Iris CLAHE enhanced regions
                cv::imwrite(out_dir + "/debug_03_iris_clahe_prev.png", diagnostics->iris_prev);
                cv::imwrite(out_dir + "/debug_03_iris_clahe_curr.png", diagnostics->iris_curr);
                
                // Vertical Sobel gradient images (normalized for 8-bit visual inspectability)
                cv::Mat grad_prev_8u, grad_curr_8u;
                cv::normalize(diagnostics->grad_prev, grad_prev_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::normalize(diagnostics->grad_curr, grad_curr_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::imwrite(out_dir + "/debug_04_sobel_prev.png", grad_prev_8u);
                cv::imwrite(out_dir + "/debug_04_sobel_curr.png", grad_curr_8u);
                
                std::cout << "\n[DEBUG] Diagnostic intermediate images saved to '" << out_dir << "/' folder:" << std::endl;
                std::cout << "  - Cleaned Cartesian: debug_01_clean_*.png" << std::endl;
                std::cout << "  - Warp Polar:        debug_02_polar_*.png" << std::endl;
                std::cout << "  - Iris CLAHE:        debug_03_iris_clahe_*.png" << std::endl;
                std::cout << "  - Sobel Gradients:   debug_04_sobel_*.png" << std::endl;
                std::cout << "[DEBUG] Measured Shift: (" << diagnostics->shift.x << ", " << diagnostics->shift.y << ")" << std::endl;
                debug_saved = true;
            }
            
            // Write results
            out_file << sample.sequence_name << ","
                     << sample.img_prev_name << ","
                     << sample.img_curr_name << ","
                     << sample.gt_angle << ","
                     << predicted_angle << ","
                     << confidence << ","
                     << (success ? "1" : "0") << "\n";
                     
            processed_count++;
        }
        
        out_file.close();
        std::cout << "\nSuccess! Pipeline executed. Processed " << processed_count 
                  << " pairs. Results saved to " << out_csv_path << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "Pipeline failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
