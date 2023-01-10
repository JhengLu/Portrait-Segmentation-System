#ifndef PROCESSFRAME_H
#define PROCESSFRAME_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <memory>

#include <QFrame>
#include <QTimer>
#include <QLabel>

using std::vector, std::string;

namespace Ui {
class ProcessFrame;
}

class ProcessFrame : public QFrame, public std::enable_shared_from_this<ProcessFrame>
{
    Q_OBJECT

public:
    explicit ProcessFrame(QWidget *parent, const QString& type, const QString& fileName);
    ~ProcessFrame();

    void showImage(QLabel *label, const cv::Mat& _image);
    void processImage(const QString& fileName);
    void processVideo(const QString& fileName);
    void processWebcam(const QString& fileName);
    void ProcessFrame::threadVideo();

private:
    Ui::ProcessFrame *ui;

    QString type;
    QString fileName;
    QString backgroundPath;
    QString outputPath;
    QString mattePath;
    QTimer *timer;
    cv::VideoCapture capture, capture2;
    cv::Mat origin, after;
    vector<string> tmpFiles;
    bool finished = false;
    float tpf = -1;
};

#endif // PROCESSFRAME_H
