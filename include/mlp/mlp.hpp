// Created by Chengfu Zou on 2023/06/17.
// Implement Multi-Layer Perceptron

#ifndef MLP_MLP_HPP
#define MLP_MLP_HPP

// std
#include <fstream>
#include <memory>
#include <random>
// 3rd party
#include <fmt/color.h>
#include <fmt/core.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
// project
#include "mlp/function.hpp"

namespace mlp {
using namespace Eigen;

/**
 * @brief Fully connected layer
 * */
template <typename T, int input_size, int output_size>
class FCLayer {
 public:
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  FCLayer() {
    // Allocate memory for weights and bias
    weights_ = std::make_unique<MatrixXT>(output_size, input_size);
    bias_ = std::make_unique<MatrixXT>(output_size, 1);
    // Initialize weights and bias randomly
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < input_size * output_size; ++i) {
      (*weights_).data()[i] = dist(engine);
    }
    for (int i = 0; i < output_size; ++i) {
      (*bias_).data()[i] = dist(engine);
    }
  }

  ~FCLayer() = default;

  MatrixXT operator()(const MatrixXT &input) const {
    // Replicate bias to match the size of input when training in batch
    auto bias = (*bias_).replicate(1, input.cols());
    return (*weights_) * input + bias;
  }
  MatrixXT &weights() { return *weights_; }
  MatrixXT &bias() { return *bias_; }

 private:
  std::unique_ptr<MatrixXT> weights_;
  std::unique_ptr<MatrixXT> bias_;
};

/**
 * @brief Multi-layer perceptron (MLP) AKA fully-connected neural network
 * consisting of 3 layers
 * */
template <typename T, int input_size, int output_size>
class MLP {
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  MLP() {
    // Check data type
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, int>::value ||
                      std::is_same<T, unsigned char>::value,
                  "Unsupported data type");
  }
  ~MLP() = default;

  /**
   * @brief Forward propagation
   * */
  MatrixXT forward(const MatrixXT &input) const {
    // Forward propagation
    auto z1 = fc1_(input);
    auto a1 = function::relu<T>(z1);
    auto z2 = fc2_(a1);
    auto a2 = function::relu<T>(z2);
    auto z3 = fc3_(a2);
    auto a3 = function::softmax<T>(z3);
    return a3;
  }
  /**
   * @brief Training MLP
   * @return Loss
   * */
  double train(int epoch, const std::vector<cv::Mat> &train_data,
               const std::vector<cv::Mat> &test_data,
               const std::vector<std::vector<uint8_t>> &train_labels,
               const std::vector<std::vector<uint8_t>> &test_labels,
               int batch_size, double learning_rate) {
    double loss = DBL_MAX;
    double lr = learning_rate;
    for (int i = 0; i < epoch; ++i) {
      fmt::print(fg(fmt::color::white), "Epoch {}/{}:\n", i + 1, epoch);

      // Shuffle dataset
      std::vector<int> indices(train_data.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::random_device seed_gen;
      std::mt19937 engine(seed_gen());
      std::shuffle(indices.begin(), indices.end(), engine);

      // Create training batch
      MatrixXT train_data_batch(input_size, batch_size);
      MatrixXT labels_batch(output_size, batch_size);
      for (int j = 0; j < batch_size; ++j) {
        // Eigen::Map is used to avoid copying cv::Mat to Eigen::MatrixXT
        int index = indices[j];
        Eigen::Map<const MatrixXT> input(train_data[index].ptr<T>(), input_size,
                                         1);
        train_data_batch.col(j) = input;
        for (int k = 0; k < output_size; ++k) {
          labels_batch(k, j) = static_cast<T>(train_labels[index][k]);
        }
      }
      // Train
      loss = function::cross_entropy<T>(this->forward(train_data_batch),
                                        labels_batch);
      fmt::print(fg(fmt::color::white), "  Loss: {}\n", loss);
      this->back_propagation(train_data_batch, labels_batch, lr);

      // Calculate accuracy every 10 epochs
      if ((i + 1) % 10 == 0) {
        this->test(test_data, test_labels);
      }
    }
    return loss;
  }

  /**
   * @brief Training MLP by step
   * @return Loss
   * */
  double train_by_step(const std::vector<cv::Mat> &train_data,
                       const std::vector<std::vector<uint8_t>> &train_labels,
                       int batch_size, double learning_rate) {
    double loss = this->train(1, train_data, {}, train_labels, {}, batch_size,
                              learning_rate);
    return loss;
  }

  /**
   * @brief Test, loop over all test data and calculate accuracy, no batch
   * @return Accuracy
   * */
  double test(const std::vector<cv::Mat> &test_data,
              const std::vector<std::vector<uint8_t>> &labels) {
    int correct = 0;
    for (int i = 0; i < test_data.size(); ++i) {
      Eigen::Map<const MatrixXT> input(test_data[i].ptr<T>(), input_size, 1);
      MatrixXT output = this->forward(input);
      int prediction = -1;
      output.col(0).maxCoeff(&prediction);
      int label = -1;
      for (int j = 0; j < output_size; ++j) {
        if (labels[i][j] == 1) {
          label = j;
          break;
        }
      }
      if (prediction == label) {
        ++correct;
      }
    }
    double accuracy = static_cast<double>(correct) / test_data.size() * 100;
    fmt::print(fg(fmt::color::white), "Accuracy: {}/{} ({:.2f}%)\n", correct,
               test_data.size(), accuracy);
    return accuracy;
  }

  /**
   * @brief Read weights from file
   * */
  void read_weights(const std::string &path) {
    std::ifstream file(path);
    if (file.is_open()) {
      auto &fc1_weights = fc1_.weights();
      for (int row = 0; row < fc1_weights.rows(); ++row) {
        for (int col = 0; col < fc1_weights.cols(); ++col) {
          file >> fc1_weights(row, col);
        }
      }
      auto &fc1_bias = fc1_.bias();
      for (int row = 0; row < fc1_bias.rows(); ++row) {
        for (int col = 0; col < fc1_bias.cols(); ++col) {
          file >> fc1_bias(row, col);
        }
      }
      auto &fc2_weights = fc2_.weights();
      for (int row = 0; row < fc2_weights.rows(); ++row) {
        for (int col = 0; col < fc2_weights.cols(); ++col) {
          file >> fc2_weights(row, col);
        }
      }
      auto &fc2_bias = fc2_.bias();
      for (int row = 0; row < fc2_bias.rows(); ++row) {
        for (int col = 0; col < fc2_bias.cols(); ++col) {
          file >> fc2_bias(row, col);
        }
      }
      auto &fc3_weights = fc3_.weights();
      for (int row = 0; row < fc3_weights.rows(); ++row) {
        for (int col = 0; col < fc3_weights.cols(); ++col) {
          file >> fc3_weights(row, col);
        }
      }
      auto &fc3_bias = fc3_.bias();
      for (int row = 0; row < fc3_bias.rows(); ++row) {
        for (int col = 0; col < fc3_bias.cols(); ++col) {
          file >> fc3_bias(row, col);
        }
      }
      fmt::print(fg(fmt::color::green), "Read weights from {}\n", path);
      file.close();
    }
  }

  /**
   * @brief Save weights to file
   * */
  void save_weights(const std::string &path) {
    std::ofstream file(path);
    if (file.is_open()) {
      file << fc1_.weights();
      file << fc1_.bias();
      file << fc2_.weights();
      file << fc2_.bias();
      file << fc3_.weights();
      file << fc3_.bias();
      fmt::print(fg(fmt::color::green), "Saved weights to {}\n", path);
      file.close();
    }
  }

  /**
   * @brief Get corresponding data type in OpenCV
   * */
  int type() const { return cv_type_impl(static_cast<T*>(nullptr)); }

 private:
  /**
   * @brief Back propagation
   * */
  void back_propagation(const MatrixXT &input, const MatrixXT &label,
                        double learning_rate) {
    // Forward propagation
    MatrixXT z1 = fc1_(input);
    MatrixXT a1 = function::relu<T>(z1);
    MatrixXT z2 = fc2_(a1);
    MatrixXT a2 = function::relu<T>(z2);
    MatrixXT z3 = fc3_(a2);
    MatrixXT a3 = function::softmax<T>(z3);

    // Compute gradients
    MatrixXT &fc1_weights = fc1_.weights();
    MatrixXT &fc2_weights = fc2_.weights();
    MatrixXT &fc3_weights = fc3_.weights();
    MatrixXT &fc1_bias = fc1_.bias();
    MatrixXT &fc2_bias = fc2_.bias();
    MatrixXT &fc3_bias = fc3_.bias();

    MatrixXT grad_z3 = a3 - label;
    MatrixXT grad_W3 = a2 * grad_z3.transpose();
    ;
    MatrixXT grad_b3 = grad_z3.rowwise().sum();
    MatrixXT grad_a2 = fc3_weights.transpose() * grad_z3;
    MatrixXT grad_z2 =
        grad_a2.array() * function::relu_derivative<T>(z2).array();

    MatrixXT grad_W2 = a1 * grad_z2.transpose();
    MatrixXT grad_b2 = grad_z2.rowwise().sum();
    MatrixXT grad_a1 = fc2_weights.transpose() * grad_z2;
    MatrixXT grad_z1 =
        grad_a1.array() * function::relu_derivative<T>(z1).array();

    MatrixXT grad_W1 = input * grad_z1.transpose();
    MatrixXT grad_b1 = grad_z1.rowwise().sum();

    // Update weights and bias
    fc1_weights -= learning_rate * grad_W1.transpose();
    fc1_bias -= learning_rate * grad_b1;
    fc2_weights -= learning_rate * grad_W2.transpose();
    fc2_bias -= learning_rate * grad_b2;
    fc3_weights -= learning_rate * grad_W3.transpose();
    fc3_bias -= learning_rate * grad_b3;
  }

  static int cv_type_impl(const float*) { return CV_32F; }
  static int cv_type_impl(const int*) { return CV_32S; }
  static int cv_type_impl(const double*) { return CV_64F; }
  static int cv_type_impl(const unsigned char*) { return CV_8U; }

 private:
  FCLayer<T, input_size, 128> fc1_;
  FCLayer<T, 128, 64> fc2_;
  FCLayer<T, 64, output_size> fc3_;
};
}  // namespace mlp
#endif  // MLP_MLP_HPP