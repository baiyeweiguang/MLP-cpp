// Created by Chengfu Zou on 2023/06/17.
// Implement some common math functions used in MLP

#ifndef MLP_FUNCTION_HPP
#define MLP_FUNCTION_HPP
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace mlp::function {
using namespace Eigen;
/**
 * @brief Rectified linear unit (ReLU)
 * */
template <typename T>
auto relu(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input)
    -> Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
  return input.unaryExpr([&](T x) { return x > 0 ? x : 0; });
}
/**
 * @brief Derivative of ReLU
 * */
template <typename T>
auto relu_derivative(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input)
    -> Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
  return input.unaryExpr([&](T x) { return x > 0 ? static_cast<T>(1.0) : 0; });
} /**
 * @brief Softmax
 * */
template <typename T>
auto softmax(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input)
    -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
  auto exp = input.array().exp();
  return exp.array().rowwise() / exp.array().colwise().sum();
}
/**
 * @brief Cross entropy
 * @param input The output of the last layer
 * @param label The ground truth label
 * */
template <typename T>
double cross_entropy(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input,
                const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &label) {
  const T eps = static_cast<T>(1e-7);
  // Restrict input to [eps, 1 - eps], the probability of each class cant be 0 or 1
  auto clipped_input = (input.array().max(eps)).min(1.0 - eps);
  return -(label.array() * clipped_input.array().log()).sum();
}

/**
 * @brief One-hot encoding
 * @param label The ground truth label
 * @param num_classes The number of classes
 * */
static std::vector<int> one_hot(const int &label, const int &num_classes = 10) {
  std::vector<int> one_hot(num_classes, 0);
  one_hot[label] = 1;
  return one_hot;
}

/**
 * @brief Histogram of oriented gradients (HOG)
 * */
static cv::Mat hog(const cv::Mat &src) {
  cv::HOGDescriptor hog(cv::Size(28, 28), cv::Size(14, 14), cv::Size(7, 7),
                        cv::Size(7, 7), 9);
  std::vector<float> descriptors;
  hog.compute(src, descriptors);
  cv::Mat hog_mat(descriptors, true);
  return hog_mat;
}
}  // namespace mlp::function
#endif  // MLP_FUNCTION_HPP