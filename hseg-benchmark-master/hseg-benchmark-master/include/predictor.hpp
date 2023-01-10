//
// Created by Lsc2001 on 2022/3/21.
//

#ifndef LIBTORCHTEST_PREDICTOR_HPP
#define LIBTORCHTEST_PREDICTOR_HPP


#define USE_LIBTORCH
#define USE_ONNX

#include "functional.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <cassert>

#include <_torch/torch.h>

namespace seg {


using std::cout, std::endl;
using std::string, std::vector;
using std::shared_ptr, std::make_shared;
using std::initializer_list;

template<typename T>
class Predictor {
public:
    virtual T predict(const cv::Mat &image) = 0;
    virtual T predict(const T &image) = 0;
};

#ifdef USE_LIBTORCH


class TorchPredictor : public Predictor<torch::Tensor> {
private:
    shared_ptr<torch::jit::Module> module;

public:
    explicit TorchPredictor(const string &path) {
        module = make_shared<torch::jit::Module>(torch::jit::load(path));
        module->eval();
    }

    torch::Tensor predict(const cv::Mat &image) override {
        auto tensor = toTensor(image);
        vector<torch::jit::IValue> inputs;
        inputs.emplace_back(std::move(tensor));
//        auto output = module->forward(inputs).toTuple();
//        return output->elements()[0].toTensor();
        return module->forward(inputs).toTensor();
    }

    torch::Tensor predict(const torch::Tensor &image) override {
        vector<torch::jit::IValue> inputs;
        inputs.emplace_back(image);
        auto output = module->forward(inputs).toTuple();
        return output->elements()[0].toTensor();
    }
};

#endif // USE_LIBTORCH

#ifdef USE_ONNX

#endif // USE_ONNX


}

#endif //LIBTORCHTEST_PREDICTOR_HPP
