#include "mlp/data_loader.hpp"
#include "mlp/mlp.hpp"

#include <fmt/color.h>
#include <fmt/core.h>

template <typename T, int input_size, int output_size>
void random_image_detection_test(
    const mlp::Dataset::Ptr &dataset,
    const mlp::MLP<T, input_size, output_size> &mlp) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_int_distribution<> dist(0, dataset->images.size() - 1);
  while (true) {
    // Get random index
    int index = dist(engine);
    // cv::Mat to Eigen::MatrixXd
    cv::Mat feature = mlp::function::hog(dataset->images[index]);
    feature.convertTo(feature, mlp.type());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> input(
        feature.ptr<T>(), 324, 1);
    // Forward propagation
    auto output = mlp.forward(input);
    // Get prediction
    int prediction = -1;
    output.col(0).maxCoeff(&prediction);
    // Show result
    cv::Mat show_image = dataset->images[index].clone();
    cv::resize(show_image, show_image, cv::Size(128, 128));
    cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2BGR);
    cv::putText(show_image, std::to_string(prediction), cv::Point(30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 120, 0), 2,
                cv::LINE_AA);
    cv::imshow("exit:'q'", show_image);
    auto key = cv::waitKey(0);
    if (key == 'q') {
      break;
    }
  }
}

int main(int argc, char** argv) {
  using namespace mlp;
  //324: input layer size
  //10: output layer size
  MLP<double, 324, 10> mlp;
  try {
    // Load MNIST dataset
    DataLoader train_data_loader("../dataset/train-images.idx3-ubyte",
                                 "../dataset/train-labels.idx1-ubyte");
    DataLoader test_data_loader("../dataset/t10k-images.idx3-ubyte",
                                "../dataset/t10k-labels.idx1-ubyte");
    auto train_dataset = train_data_loader.load();
    auto test_dataset = test_data_loader.load();
    // Extract HOG features
    std::vector<cv::Mat> train_features;
    std::vector<cv::Mat> test_features;
    for (const auto &image : train_dataset->images) {
      cv::Mat features = function::hog(image);
      features.convertTo(features, mlp.type());
      train_features.emplace_back(features);
    }
    for (const auto &image : test_dataset->images) {
      cv::Mat features = function::hog(image);
      features.convertTo(features, mlp.type());
      test_features.emplace_back(features);
      features.type();
    }

    std::cout << "Number of training images: " << train_dataset->images.size()
              << std::endl;
    std::cout << "Number of test images: " << test_dataset->images.size()
              << std::endl;
    // Train MLP
    mlp.train(150, train_features, test_features, train_dataset->labels,
              test_dataset->labels, 1000, 0.0001);
    mlp.save_weights("./wth.txt");
    //    mlp.read_weights("./wth.txt");
    //    mlp.test(test_features, test_dataset->labels);

    // Test MLP
    random_image_detection_test(test_dataset, mlp);

  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  return 0;
}
