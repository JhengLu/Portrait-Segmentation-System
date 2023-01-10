//
// Created by Lsc2001 on 2022/3/23.
//

#ifndef LIBTORCHTEST_DATASET_HPP
#define LIBTORCHTEST_DATASET_HPP

#include "functional.hpp"

#include <opencv2/opencv.hpp>
#include <_torch/torch.h>

#include <cstdlib>

#include <iostream>
#include <utility>
#include <vector>
#include <filesystem>

using std::cerr, std::endl;
using std::vector, std::string;
namespace fs { using namespace std::filesystem; }

template <typename T>
class Dataset {
public:
    virtual size_t size() = 0;
};

#define PictureDataset ImageDataset

class ImageDataset : public Dataset<torch::Tensor> {
protected:
    string root;
    vector<string> imageNames;

    void traverse(const string& path) {
        fs::directory_entry entry(path);
        if (entry.status().type() != fs::file_type::directory) return;

        fs::directory_iterator fileList(path);
        for (auto& file : fileList) {
            if (file.status().type() == fs::file_type::directory) {
                traverse(file.path().string());
            }
            else imageNames.push_back(file.path().string());
        }
    }

public:
    explicit ImageDataset(string _root): root(std::move(_root)) {
        if (!fs::exists(root)) {
            cerr << "Path not exist!" << endl;
            exit(1);
        }
        fs::directory_entry entry(root);
        if (entry.status().type() != fs::file_type::directory) {
            cerr << "Path is not a directory!" << endl;
            exit(1);
        }
        traverse(root);
    }

    size_t size() override { return imageNames.size(); }
    void checkBound(size_t pos) {
        if (pos >= size()) {
            cerr << "Pos out of bound!" << endl;
            exit(1);
        }
    }

    virtual torch::Tensor get(size_t pos) {
        return toTensor(getMat(pos));
    }

    cv::Mat getMat(size_t pos) {
        checkBound(pos);
        return cv::imread(imageNames[pos]);
    }

    string getPath(size_t pos) {
        checkBound(pos);
        return imageNames[pos];
    }

    torch::Tensor operator[](size_t pos) { return get(pos); }
};

class LabelDataset : public ImageDataset {
private:
    int num_classes;

public:
    static torch::Tensor toLabel(const cv::Mat& label) {
        auto tensor = torch::from_blob(label.data, {1, label.rows, label.cols, 1}, torch::kU8)
                .to(torch::kInt64)
                .permute({0, 3, 1, 2});
        // I'm lazy
        tensor[tensor <= 127] = 0;
        tensor[tensor >= 128] = 1;
        return tensor;
    }

    explicit LabelDataset(string _root, int num_classes):
        ImageDataset(std::move(_root)),
        num_classes(num_classes) {}

    torch::Tensor get(size_t pos) override {
        if (pos >= size()) {
            cerr << "Pos out of bound!" << endl;
            exit(1);
        }
        cv::Mat image = cv::imread(imageNames[pos]);
        return toLabel(image);
    }
};

class VideoDataset : public Dataset<torch::Tensor> {
private:
    string path;
    cv::VideoCapture cap;
    size_t sz, curPos = 0;

public:
    explicit VideoDataset(string _path): path(std::move(_path)) {
        cap.open(path);
        if (!cap.isOpened()) {
            cerr << "Check the video path!" << endl;
            exit(0);
        }
        sz = cap.get(cv::CAP_PROP_FRAME_COUNT);
    }

    size_t size() override { return sz; }

    torch::Tensor next() {
        if (curPos++ >= sz) {
            cerr << "The last frame has been read!" << endl;
            exit(0);
        }
        cv::Mat frame;
        cap.read(frame);
        return toTensor(frame);
    }
};

#endif //LIBTORCHTEST_DATASET_HPP
