#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "algorithms/PolarCrossCorrelation.hpp"
#include <cmath>

// Test Fixture to access protected members
class PolarCrossCorrelationTest : public ::testing::Test, public PolarCrossCorrelation {
public:
    PolarCrossCorrelationTest() : PolarCrossCorrelation(80, 1440, false) {}
    
    // Wrapper methods to expose protected members for testing
    cv::Mat testConvertToPolar(const cv::Mat& src) {
        return this->convertToPolar(src);
    }
    
    std::pair<cv::Mat, cv::Mat> testRemoveGlints(const cv::Mat& src) {
        return this->removeGlints(src);
    }
    
    double testCalculateMaskedNCC(const cv::Mat& prev, const cv::Mat& curr, 
                                  const cv::Mat& mask_prev, const cv::Mat& mask_curr, 
                                  int max_shift, double& best_peak) {
        return this->calculateMaskedNCC(prev, curr, mask_prev, mask_curr, max_shift, best_peak);
    }
};

// 1. Test convertToPolar Dimensions Check
TEST_F(PolarCrossCorrelationTest, ConvertToPolarDimensions) {
    int img_size = 400;
    cv::Mat dummy_cartesian = cv::Mat::zeros(img_size, img_size, CV_8UC1);
    
    cv::Mat polar = testConvertToPolar(dummy_cartesian);
    
    // Polar image should have width = angular_bins_ and height = radial_bins_
    EXPECT_EQ(polar.cols, 1440);
    EXPECT_EQ(polar.rows, 80);
    EXPECT_EQ(polar.type(), CV_8UC1);
}

// 2. Test removeGlints Mask Generation
TEST_F(PolarCrossCorrelationTest, RemoveGlintsMaskGeneration) {
    int img_size = 400;
    cv::Mat synthetic_img = cv::Mat::zeros(img_size, img_size, CV_8UC1);
    
    // Draw a dark gray background
    synthetic_img.setTo(50);
    
    // Draw a pure white glint circle in the middle
    cv::Point center(200, 200);
    int radius = 20;
    cv::circle(synthetic_img, center, radius, cv::Scalar(255), -1);
    
    auto [cleaned, mask] = testRemoveGlints(synthetic_img);
    
    // Mask should be 0 at the glint center, and 1 at the background
    EXPECT_EQ(mask.at<uchar>(center), 0);
    EXPECT_EQ(mask.at<uchar>(10, 10), 1);
    
    // Cleaned image should have inpainted the glint, so the center shouldn't be 255 anymore
    EXPECT_LT(cleaned.at<uchar>(center), 255);
}

// 3. Test calculateMaskedNCC pure shift verification
TEST_F(PolarCrossCorrelationTest, CalculateMaskedNCCExactShift) {
    // Create two 1D floating point rows of size 1440 (angular bins)
    cv::Mat prev_row = cv::Mat::zeros(1, 1440, CV_32FC1);
    cv::Mat curr_row = cv::Mat::zeros(1, 1440, CV_32FC1);
    
    // Create masks (all valid = 1)
    cv::Mat mask_prev = cv::Mat::ones(1, 1440, CV_8UC1);
    cv::Mat mask_curr = cv::Mat::ones(1, 1440, CV_8UC1);
    
    // Populate prev_row with a distinct pattern (e.g., a localized peak or square wave)
    for (int i = 500; i < 550; ++i) {
        prev_row.at<float>(0, i) = 100.0f;
    }
    
    // Define a true integer shift
    int true_shift = 15;
    
    // Shift the pattern into curr_row (handling circular wrapping if needed, but here simple shift is fine)
    for (int i = 0; i < 1440; ++i) {
        int src_i = (i - true_shift + 1440) % 1440;
        curr_row.at<float>(0, i) = prev_row.at<float>(0, src_i);
    }
    
    double peak_response = 0;
    int max_search_shift = 45; // 45 degrees * 4 bins/deg = 180 bins
    double detected_shift = testCalculateMaskedNCC(prev_row, curr_row, mask_prev, mask_curr, max_search_shift, peak_response);
    
    EXPECT_DOUBLE_EQ(detected_shift, static_cast<double>(true_shift));
    EXPECT_GT(peak_response, 0.99); // Confidence should be perfect for identical arrays
}

// 4. End-to-End calculateTorsion (Optional basic sanity check)
TEST_F(PolarCrossCorrelationTest, EndToEndRotationNoTranslation) {
    // Create a 400x400 synthetic iris-like image.
    // We use dense uniform noise within an iris ring so that every CLAHE tile
    // has rich intensity variation, making the masked NCC reliable.
    cv::Mat img1 = cv::Mat::zeros(400, 400, CV_8UC1);
    cv::Point center(200, 200);

    // Fill with deterministic noise in [20, 120] — all values below the
    // removeGlints threshold of 140, so nothing gets inpainted.
    cv::theRNG().state = 123456789ULL;
    cv::randu(img1, cv::Scalar(20), cv::Scalar(120));

    // Apply iris annulus mask: keep only the ring from radius 30 to 75 pixels.
    cv::Mat iris_ring = cv::Mat::zeros(400, 400, CV_8UC1);
    cv::circle(iris_ring, center, 75, cv::Scalar(255), -1); // outer boundary
    cv::circle(iris_ring, center, 30, cv::Scalar(0),   -1); // inner (pupil)
    img1.setTo(0, iris_ring == 0);

    // Create a rotated version (+5 degrees in OpenCV convention = 5° clockwise visually).
    double true_rotation_deg = 5.0;
    cv::Mat M = cv::getRotationMatrix2D(center, true_rotation_deg, 1.0);
    cv::Mat img2;
    cv::warpAffine(img1, img2, M, img1.size(), cv::INTER_LINEAR);

    // Run the full algorithm.
    TorsionResult result = calculateTorsion(img1, img2, false);

    // The calculated angle should match the true rotation within 1 degree.
    // (Small discretization error is expected due to warpAffine interpolation.)
    EXPECT_NEAR(result.angle, true_rotation_deg, 1.0);
    EXPECT_TRUE(result.success);
}
