#include "DataLoader.hpp"
#include "TorsionDetector.hpp"
#include "algorithms/TorsionAlgorithm.hpp"
#include "algorithms/PolarCrossCorrelation.hpp"
#include "utils/PerformanceTracker.hpp"
#include "utils/TorsionVisualizer.hpp"
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
        
        // 2. Initialize TorsionDetector with Polar Cross-Correlation Strategy (1440 angular bins)
        // Default is use_fft = false (Masked Spatial Correlation)
        auto polar_algo = std::make_unique<PolarCrossCorrelation>(80, 1440);
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
        out_file << "sequence,img_prev,img_curr,gt_angle,algo_angle,confidence,success,runtime_ms\n";
        
        // 4. Iterate through samples lazily
        size_t processed_count = 0;
        bool debug_saved = false;
        PerformanceTracker tracker;
        
        while (loader.hasNext()) {
            TorsionSample sample = loader.next();
            
            // Request diagnostics only for the first frame to save debug overlays
            bool request_diag = !debug_saved;
            
            tracker.start();
            auto [predicted_angle, confidence, success, diagnostics] = detector.process(sample.img_prev, sample.img_curr, request_diag);
            double elapsed_ms = tracker.stop();
            
            // Save one pair of diagnostic images inside the dynamic output directory
            if (diagnostics && !debug_saved) {
                TorsionVisualizer::saveDiagnostics(*diagnostics, out_dir);
                debug_saved = true;
            }
            
            // Write results with standard float precision
            out_file << std::fixed << std::setprecision(6)
                     << sample.sequence_name << ","
                     << sample.img_prev_name << ","
                     << sample.img_curr_name << ","
                     << sample.gt_angle << ","
                     << predicted_angle << ","
                     << confidence << ","
                     << (success ? "1" : "0") << ","
                     << elapsed_ms << "\n";
                     
            processed_count++;
        }
        
        out_file.close();
        std::cout << "\nSuccess! Pipeline executed. Processed " << processed_count 
                  << " pairs. Results saved to " << out_csv_path << std::endl;
                  
        std::cout << "\n=== Algorithm Latency Stats ===" << std::endl;
        std::cout << "Mean:   " << tracker.getMean() << " ms" << std::endl;
        std::cout << "Min:    " << tracker.getMin() << " ms" << std::endl;
        std::cout << "Max:    " << tracker.getMax() << " ms" << std::endl;
        std::cout << "StdDev: " << tracker.getStdDev() << " ms" << std::endl;
                  
        extern PerformanceTracker g_algo_tracker;
        g_algo_tracker.printReport();
                  
                  
    } catch (const std::exception& e) {
        std::cerr << "Pipeline failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
