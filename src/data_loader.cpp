#include "mlp/data_loader.hpp"
#include "mlp/function.hpp"

#include <fstream>
#include <random>

namespace mlp {
DataLoader::DataLoader(const std::string &image_filename,
                       const std::string &label_filename)
    : image_filename_(image_filename), label_filename_(label_filename) {}

uint32_t DataLoader::read_uint32(std::ifstream &file) {
  uint32_t result;
  file.read(reinterpret_cast<char*>(&result), 4);
  return __builtin_bswap32(result);
}

Dataset::Ptr DataLoader::load() {
  std::ifstream image_file(image_filename_, std::ios::binary);
  std::ifstream label_file(label_filename_, std::ios::binary);

  if (!image_file.is_open() || !label_file.is_open()) {
    throw std::runtime_error("Error: Unable to open MNIST dataset files.");
  }

  if (read_uint32(image_file) != 0x803) {
    throw std::runtime_error("Error: Invalid MNIST image file.");
  }

  if (read_uint32(label_file) != 0x801) {
    throw std::runtime_error("Error: Invalid MNIST label file.");
  }

  uint32_t image_count = read_uint32(image_file);
  uint32_t label_count = read_uint32(label_file);

  if (image_count != label_count) {
    throw std::runtime_error("Error: Image count and label count mismatch.");
  }

  uint32_t rows = read_uint32(image_file);
  uint32_t cols = read_uint32(image_file);

  auto dataset = std::make_unique<Dataset>();
  dataset->images.resize(image_count);
  dataset->labels.resize(label_count, std::vector<uint8_t>(10));

  for (uint32_t i = 0; i < image_count; ++i) {
    cv::Mat image(rows, cols, CV_8UC1);
    image_file.read(reinterpret_cast<char*>(image.data), rows * cols);
    dataset->images[i] = image;

    uint8_t label;
    label_file.read(reinterpret_cast<char*>(&label), 1);
    dataset->labels[i][label] = 1;
  }

  return dataset;
}

}  // namespace mlp