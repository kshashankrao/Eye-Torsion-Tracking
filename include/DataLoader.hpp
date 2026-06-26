#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct TorsionSample {
    cv::Mat img_prev;          // Grayscale image frame t-1
    cv::Mat img_curr;          // Grayscale image frame t
    double gt_angle;           // Ground truth rotation in degrees
    std::string img_prev_name; // Filename for t-1
    std::string img_curr_name; // Filename for t
    std::string sequence_name; // Name of sequence folder (e.g. "seq_00")
};

class DataLoader {
public:
    explicit DataLoader(const std::string& config_path);

    // Iterator interface
    bool hasNext() const;
    TorsionSample next();
    void reset();

    // Utility getters
    size_t getTotalSamples() const;
    size_t getCurrentIndex() const;

private:
    struct SampleMetadata {
        std::string seq_name;
        std::string img_prev_path;
        std::string img_curr_path;
        double gt_angle;
        std::string img_prev_name;
        std::string img_curr_name;
    };

    std::string processed_dir_;
    std::vector<std::string> sequence_names_;
    std::vector<SampleMetadata> samples_;
    size_t current_idx_ = 0;

    void parseConfig(const std::string& config_path);
    void loadSequencesMetadata();
    void parseCSV(const std::string& csv_path, const std::string& seq_name, const std::string& seq_path);
};
