//
// Created by Lsc2001 on 2022/3/24.
//

#include "processing.h"

#include <cmdline.h>
#include <functional.hpp>
#include <predictor.hpp>
#include <utils.hpp>
#include <dataset.hpp>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>


using std::cout, std::cerr, std::endl;
using std::string, std::shared_ptr;
using seg::TorchPredictor;

shared_ptr<TorchPredictor> predictor;
namespace F { using namespace torch::nn::functional; }

void setPredictor(const string& modelPath) {
    predictor = std::make_shared<TorchPredictor>(modelPath);
}

clock_t predict(const cv::Mat& image, cv::Mat& output) {
    clock_t start = clock();

    int H = image.rows, W = image.cols;
    output = toMat<uint8_t>(
            predictor->predict(image)
                    .squeeze()
                    .argmax(0)
                    .to(torch::kU8)
                    .mul(255)
    );
    cv::resize(output, output, cv::Size(W, H), cv::INTER_AREA);

    clock_t end = clock();
    return end - start;
}

cv::Mat tryFuse(cv::Mat& image, const string& backgroundPath, cv::Mat& matte) {
    cv::Mat background;
    if (!backgroundPath.empty()) {
        background = cv::imread(backgroundPath);
        if (image.size != background.size) {
            cv::resize(image, image, cv::Size(background.cols, background.rows));
            cv::resize(matte, matte, cv::Size(background.cols, background.rows));
        }
        return fuse(image, background, repeatChannel(matte, 3));
    }
    return matte;
}

void handleImage(const string& path, const string& backgroundPath, const string& outputPath) {
    cv::Mat image = cv::imread(path);
//    cv::resize(image, image, cv::Size(640, 480));

    cv::Mat output;
    long t = predict(image, output);

    if (!outputPath.empty()) {
        output = tryFuse(image, backgroundPath, output);
        cv::imwrite(outputPath, output);
    }
    cout << "Preprocessing and inference time: " << t << "ms" << endl;
}

void handleVideo(const string& path, const string& mattePath,
                 const string& backgroundPath, const string& outputPath) {

    cv::VideoCapture capture(path);
    cv::VideoCapture mCapture;
    if (!mattePath.empty()) mCapture.open(mattePath);

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(cv::CAP_PROP_FPS);
    int numFrame = capture.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer;
    if (!outputPath.empty())
        writer.open(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    cv::Mat frame, matte;
    torch::Tensor label;
    ConfusionMatrix confusion(2);
    clock_t t = 0;
    int cnt = 0;

    while (true) {
        capture >> frame;
        if (!mattePath.empty()) {
            mCapture >> matte;
            label = LabelDataset::toLabel(matte);
        }

        if (frame.empty()) break;
        int H = frame.rows, W = frame.cols;

        clock_t start = clock();

        auto output = predictor->predict(frame)
                                .squeeze(0).argmax(0).to(torch::kU8);
        output *= 255;
        clock_t mid = clock();

        auto outputMat = toMat<uint8_t>(output);
        cv::resize(outputMat, outputMat, cv::Size(W, H), cv::INTER_AREA);

        clock_t end = clock();
        t += end - start;

        if (!mattePath.empty())
            confusion.update(label.flatten(), output.flatten());

        if (writer.isOpened()) {
            outputMat = tryFuse(frame, backgroundPath, outputMat);
            outputMat = repeatChannel(outputMat, 3);
            writer.write(outputMat);
        }
        printf("%.2f%% finished.\n", (float)cnt++ / numFrame * 100);
    }

    writer.release();
    if (!mattePath.empty())
        printf("Mean IOU: %.2f%%\n", confusion.getMIOU() * 100);
    printf("Preprocessing and inference time: %.2fms/frame\n", (float)t / numFrame);
}


void handleImageFolder(const string& imageFolderPath, const string& matteFolderPath,
                                     const string& backgroundPath, const string& outputPath) {

    auto images = std::make_shared<ImageDataset>(imageFolderPath);
    shared_ptr<LabelDataset> labels = nullptr;
    if (!matteFolderPath.empty())
        labels = std::make_shared<LabelDataset>(matteFolderPath, 2);

    ConfusionMatrix confusion(2);
    size_t total = images->size();
    clock_t t = 0;

    for (int i = 0; i < total; i++) {
        string name = absolutePath(images->getPath(i), imageFolderPath);
        cv::Mat image = images->getMat(i);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        torch::Tensor label;
        if (!matteFolderPath.empty()) label = labels->get(i);

        clock_t start = clock();
        auto output = predictor->predict(image)
                .to(torch::kU8)
                .squeeze()
                .argmax(0);
        clock_t end = clock();
        t += end - start;
        if (!matteFolderPath.empty())
            confusion.update(label.flatten(), output.flatten());

        if (!outputPath.empty()) {
            auto newPath = joinPath(outputPath, "processed_" + name);
            cv::Mat outputMat = toMat<uint8_t>(output);
            outputMat = tryFuse(image, backgroundPath, outputMat);

            cv::imwrite(newPath, outputMat);
        }
    }
    printf("Mean IOU: %.2f%%\n", confusion.getMIOU() * 100);
    printf("Preprocessing and inference time: %.2fms/frame\n", (float)t / total);
}

