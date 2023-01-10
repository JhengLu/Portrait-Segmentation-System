//
// Created by Lsc2001 on 2022/3/24.
//

#ifndef HSEGBENCHMARK_PROCESSING_H
#define HSEGBENCHMARK_PROCESSING_H

#include <_torch/torch.h>
#include <opencv2/opencv.hpp>

#include <string>
using std::string;

void setPredictor(const string& modelPath);
clock_t predict(const cv::Mat& image, cv::Mat& output);
void handleImage(const string& path, const string& backgroundPath, const string& outputPath);
void handleVideo(const string& path, const string& mattePath,
                 const string& backgroundPath, const string& outputPath);
void handleImageFolder(const string& imageFolderPath, const string& matteFolderPath,
                     const string& backgroundPath, const string& outputPath);

#endif //HSEGBENCHMARK_PROCESSING_H
