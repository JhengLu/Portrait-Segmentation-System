#include "processing.h"

#include <cmdline.h>
#include <_torch/torch.h>

#include <iostream>
using std::cout, std::cerr, std::endl;
using std::string;

int main(int argc, char* argv[]) {
    cmdline::parser parser;
    parser.add<string>("model-path", 0, "path of model", true);
    parser.add<string>("image-path", 0, "", false, "");
    parser.add<string>("video-path", 0, "path of input video", false, "");
    parser.add<string>("matte-path", 0, "", false, "");
    parser.add<string>("image-folder-path", 0, "", false, "");
    parser.add<string>("matte-folder-path", 0, "", false, "");
    parser.add<string>("background-path", 0, "", false, "");
    parser.add<string>("output-path", 0, "path of output video", true);
    parser.add<string>("type", 0, "segmentation or matting", true);
    parser.add<string>("output-type", 0, "foreground or matte", true);
    parser.parse_check(argc, argv);

    string modelPath = parser.get<string>("model-path");
    string imagePath = parser.get<string>("image-path");
    string videoPath = parser.get<string>("video-path");
    string mattePath = parser.get<string>("matte-path");
    string imageFolderPath = parser.get<string>("image-folder-path");
    string matteFolderPath = parser.get<string>("matte-folder-path");
    string backgroundPath = parser.get<string>("background-path");
    string outputPath = parser.get<string>("output-path");
    string type = parser.get<string>("type");
    string outputType = parser.get<string>("output-type");

    if (type != "segmentation" && type != "matting") {
        cerr << "Type should be segmentation or matting" << endl;
        return 0;
    }

    if (outputType != "foreground" && outputType != "matte") {
        cerr << "output-type should be foreground or matte" << endl;
        return 0;
    }

    int empty = (int)imagePath.empty() + videoPath.empty() + backgroundPath.empty();
    if (empty != 2) {
        cerr << "Please enter exactly one of inputs." << endl;
        return 0;
    }

    if (type == "matting") {
        bool flag = true;
        if (videoPath.empty())
            cerr << "Matting only supports video." << endl, flag = false;
        else if (!mattePath.empty())
            cerr << "MIOU calculation for matting haven't been implemented" << endl, flag = false;

        if (!flag) return 0;
    }

    setPredictor(modelPath);
    cout << "Start inference..." << endl;

    if (!imagePath.empty()) {
        handleImage(imagePath, backgroundPath, outputPath);
    }
    else if (!videoPath.empty()) {
        handleVideo(videoPath, mattePath, backgroundPath, outputPath);
    }
    else if (!imageFolderPath.empty()) {
        handleImageFolder(imageFolderPath, matteFolderPath, backgroundPath, outputPath);
    }

    return 0;
}
