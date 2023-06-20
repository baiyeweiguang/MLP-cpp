# MLP-cpp
This is a course design for the "Machine Learning" class in the Artificial Intelligence major at Central South University. The project consists of a implementation of MLP and BP algorithm in C++ without any provided algorithms in third-party libraries.


## Dependencies

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (Matrix and linear algebra library)
- [OpenCV](http://docs.opencv.org) (Computer Vision library)
- [fmt](http://fmtlib.net/latest/index.html) (String formatting library)

## Tested Environment

This project has been tested and confirmed to work on Ubuntu 20.04.

## Setup and Installation

Follow these steps to set up the development environment and install the necessary dependencies:

1. Update the system package list:

```bash
sudo apt-get update
```

2. Install the required dependencies:

```bash
sudo apt-get install libeigen3-dev libopencv-dev libfmt-dev
```

3. Clone the project repository:

```bash
git clone clone https://github.com/baiyeweiguang/MLP-cpp.git
cd MLP-cpp
```

4. Compile the project:

```bash
mkdir build && cd build
cmake ..
make
```

5. Run the application:

```bash
./mlp_cpp
```

## Usage

1. Create the model:

```cpp
//324: input layer size
//10: output layer size
MLP<double, 324, 10> mlp;
```

2. Train the model:

```cpp
// Train MLP
mlp.train(epoch, train_dataset, test_dataset, train_labels,
        test_labels, batch_size, learning_rate);
```

3. Save and read weights:

```cpp
mlp.save_weights("./wth.txt");
mlp.read_weights("./wth.txt");
```

4. Predict:

```cpp
Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, input_size);
Eigen::MatrixXd output = mlp.forward(input);
```


## License

This project is available under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](./LICENSE) file for more information.
