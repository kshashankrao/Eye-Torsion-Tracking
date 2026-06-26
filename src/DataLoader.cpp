#include "DataLoader.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

DataLoader::DataLoader(const std::string& config_path) {
    parseConfig(config_path);
    loadSequencesMetadata();
}

bool DataLoader::hasNext() const {
    return current_idx_ < samples_.size();
}

TorsionSample DataLoader::next() {
    if (!hasNext()) {
        throw std::out_of_range("No more samples in DataLoader iterator!");
    }

    const auto& meta = samples_[current_idx_++];
    TorsionSample sample;

    // Load images lazily from disk in grayscale
    sample.img_prev = cv::imread(meta.img_prev_path, cv::IMREAD_GRAYSCALE);
    sample.img_curr = cv::imread(meta.img_curr_path, cv::IMREAD_GRAYSCALE);

    if (sample.img_prev.empty()) {
        throw std::runtime_error("DataLoader failed to load image: " + meta.img_prev_path);
    }
    if (sample.img_curr.empty()) {
        throw std::runtime_error("DataLoader failed to load image: " + meta.img_curr_path);
    }

    sample.gt_angle = meta.gt_angle;
    sample.img_prev_name = meta.img_prev_name;
    sample.img_curr_name = meta.img_curr_name;
    sample.sequence_name = meta.seq_name;

    return sample;
}

void DataLoader::reset() {
    current_idx_ = 0;
}

size_t DataLoader::getTotalSamples() const {
    return samples_.size();
}

size_t DataLoader::getCurrentIndex() const {
    return current_idx_;
}

void DataLoader::parseConfig(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("DataLoader could not open config file: " + config_path);
    }

    json config;
    file >> config;

    // Get paths
    processed_dir_ = config["dataset"]["processed_images_dir"].get<std::string>();
    
    // Load sequences list
    if (config["dataset"].contains("sequences")) {
        for (const auto& seq : config["dataset"]["sequences"]) {
            sequence_names_.push_back(seq.get<std::string>());
        }
    } else {
        throw std::runtime_error("Config missing 'dataset.sequences' array!");
    }
}

void DataLoader::loadSequencesMetadata() {
    samples_.clear();
    for (const auto& seq_name : sequence_names_) {
        std::string seq_path = processed_dir_ + "/" + seq_name;
        std::string csv_path = seq_path + "/ground_truth.csv";
        parseCSV(csv_path, seq_name, seq_path);
    }
    std::cout << "DataLoader parsed metadata. Total samples loaded: " << samples_.size() << std::endl;
}

void DataLoader::parseCSV(const std::string& csv_path, const std::string& seq_name, const std::string& seq_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open CSV file: " << csv_path << std::endl;
        return;
    }

    std::string line;
    // Skip CSV header (img_prev,img_curr,angle)
    if (!std::getline(file, line)) {
        return;
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string prev, curr, angle_str;

        if (std::getline(ss, prev, ',') &&
            std::getline(ss, curr, ',') &&
            std::getline(ss, angle_str, ',')) {
            
            SampleMetadata meta;
            meta.seq_name = seq_name;
            meta.img_prev_path = seq_path + "/" + prev;
            meta.img_curr_path = seq_path + "/" + curr;
            meta.img_prev_name = prev;
            meta.img_curr_name = curr;
            meta.gt_angle = std::stod(angle_str);

            samples_.push_back(meta);
        }
    }
}
