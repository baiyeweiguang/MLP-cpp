// Created by Chengfu Zou on 2023/06/17.
// Implement Data Loader
#ifndef MLP_DATA_LOADER_HPP
#define MLP_DATA_LOADER_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace mlp {
/**
 * @brief MNIST dataset
 * */
struct Dataset {
  using Ptr = std::unique_ptr<Dataset>;
  std::vector<cv::Mat> images;
  std::vector<std::vector<uint8_t>> labels;
};

/**
 * @brief MNIST dataset loader
 * */
class DataLoader {
 public:
  DataLoader(const std::string &image_filename,
             const std::string &label_filename);
  Dataset::Ptr load();

 private:
  uint32_t read_uint32(std::ifstream &file);
  std::string image_filename_;
  std::string label_filename_;
};
}  // namespace mlp
#endif  // MLP_DATA_LOADER_HPP
