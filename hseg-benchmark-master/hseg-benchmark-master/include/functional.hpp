//
// Created by Lsc2001 on 2022/3/23.
//

#ifndef LIBTORCHTEST_FUNCTIONAL_HPP
#define LIBTORCHTEST_FUNCTIONAL_HPP

#include <opencv2/opencv.hpp>
#include <_torch/torch.h>

torch::Tensor normalize(const torch::Tensor& tensor,
                        const std::vector<float>& mean,
                        const std::vector<float>& std) {
    assert(mean.size() == std.size());
    auto _mean = torch::from_blob((void*)mean.data(), {(long long)mean.size()}, torch::kFloat32)
            .view({-1, 1, 1});
    auto _std = torch::from_blob((void*)std.data(), {(long long)std.size()}, torch::kFloat32)
            .view({-1, 1, 1});
    return tensor.sub_(_mean).div_(_std);
}

torch::Tensor toTensor(const cv::Mat &image) {
    assert(image.channels() == 1 || image.channels() == 3);

    torch::Tensor tensor;
    if (image.channels() == 1) {
        tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 1}, torch::kU8)
                .div(255.);
    }

    if (image.channels() == 3) {
        cv::Mat _image;
        cv::resize(image, _image, cv::Size(512, 512));
        cv::cvtColor(_image, _image, cv::COLOR_BGR2RGB);
        tensor = torch::from_blob(_image.data, {1, _image.rows, _image.cols, 3}, torch::kU8)
                .permute({0, 3, 1, 2})
                .div(255.);
        tensor = normalize(tensor, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    }
    return tensor;
}

class ToTensor {
public:
    torch::Tensor operator() (const cv::Mat &image) { return toTensor(image); }
};

#endif //LIBTORCHTEST_FUNCTIONAL_HPP
