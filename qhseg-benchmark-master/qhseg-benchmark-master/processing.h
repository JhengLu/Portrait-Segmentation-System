//
// Created by Lsc2001 on 2022/3/24.
//

#ifndef HSEGBENCHMARK_PROCESSING_H
#define HSEGBENCHMARK_PROCESSING_H

#include <_torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
using std::string, std::pair;

void setPredictor(const string& modelPath);
clock_t predict(const cv::Mat& image, cv::Mat& output);
clock_t handleImage(const string& path, const string& backgroundPath, cv::Mat& image, cv::Mat& output);
pair<float, float> handleVideo(const string& path, const string& mattePath,
                  const string& backgroundPath, const string& outputPath);
clock_t handleWebcam(cv::VideoCapture& capture,
                     cv::Mat& origin, cv::Mat& output, const string& backgroundPath);
pair<float, float> handleImageFolder(const string& imageFolderPath, const string& matteFolderPath,
                     const string& backgroundPath, const string& outputPath);

#endif //HSEGBENCHMARK_PROCESSING_H
