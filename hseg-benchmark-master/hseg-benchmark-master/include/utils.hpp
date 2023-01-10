//
// Created by Lsc2001 on 2022/3/23.
//

#ifndef LIBTORCHTEST_UTILS_HPP
#define LIBTORCHTEST_UTILS_HPP

#ifdef _WINDOWS
#define SEP '\\'
#else
#define SEP '/'
#endif

#include <_torch/torch.h>
#include <opencv2/opencv.hpp>

#include <type_traits>
#include <iostream>
#include <vector>
#include <cstdio>

using std::cout, std::endl;
using std::string, std::pair;

struct Measurements {
    torch::Tensor globalAccuracy;
    torch::Tensor accuracy;
    torch::Tensor iou;
};

class ConfusionMatrix {
private:
    int num_classes;
    torch::Tensor mat;
public:
    explicit ConfusionMatrix(int num_classes):
        num_classes(num_classes),
        mat(torch::zeros({num_classes, num_classes}, torch::kI64)) {}

    void update(const torch::Tensor& a, const torch::Tensor& b) {
        auto k = (a >= 0) & (b < num_classes);
        auto inds = num_classes * a[k].to(torch::kI64) + b[k];
        mat += torch::bincount(inds, {}, num_classes * num_classes)
                .reshape({num_classes, num_classes});
    }

    void updateByOutput(const torch::Tensor& target, const torch::Tensor& output) {
        update(target.flatten(), output.argmax(1).flatten());
    }

    Measurements computeMeasurements() {
        auto diag = torch::diag(mat);
        return {
            diag.sum() / mat.sum(),
            diag / mat.sum(1),
            diag / (mat.sum(1) + mat.sum(0) - diag)
        };
    }

    float getMIOU() {
        return (torch::diag(mat) / (mat.sum(1) + mat.sum(0) - torch::diag(mat)))
            .mean() .item() .toFloat();
    }

    void clear() { mat.zero_(); }
};

static cv::Mat fuse(const cv::Mat& front, const cv::Mat& back, const cv::Mat& matte) {
    cv::Mat ones(matte.rows, matte.cols, CV_8UC3, cv::Scalar_(1, 1, 1));
    return front.mul(matte) + back.mul(ones - matte);
}

template <typename T>
static cv::Mat toMat(const torch::Tensor& tensor) {
    int sizes[2] = {
            (int)tensor.size(0), // row
            (int)tensor.size(1), // col
    };
    // Just simply judge, don't input any bad data
    if constexpr (std::is_floating_point<T>::value) {
        return cv::Mat(2, sizes, CV_32FC1, tensor.data_ptr()).clone();
    }
    else return cv::Mat(2, sizes, CV_8UC1, tensor.data_ptr()).clone();
}

static pair<string, string> splitPath(const string& path, char sep = SEP) {
    auto p = path.find_last_of(sep);
    return {
            path.substr(0, p),
            path.substr(p + 1)
    };
}

static string joinPath(const string& root, const string& file) {
    return root + SEP + file;
}

static string absolutePath(const string& path, const string& base) {
    auto p = path.find(base);
    if (p == string::npos) return path;
    return path.substr(base.size());
}

static cv::Mat repeatChannel(const cv::Mat& image, int channels) {
    cv::Mat res;
    std::vector<cv::Mat> vec(channels);
    for (int i = 0; i < channels; i++) vec[i] = image;
    cv::merge(vec, res);
    return res;
}

#endif //LIBTORCHTEST_UTILS_HPP
