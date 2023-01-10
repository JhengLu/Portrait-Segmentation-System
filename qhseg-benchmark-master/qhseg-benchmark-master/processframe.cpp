#include "processframe.h"
#include "ui_processframe.h"
#include "processing.h"

#include <utils.hpp>
#include <filesystem>
#include <thread>

#include <QImage>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFont>
#include <QMessageBox>
#include <QFileDialog>

namespace fs { using namespace std::filesystem; }

ProcessFrame::ProcessFrame(QWidget *parent, const QString& type, const QString& fileName) :
    QFrame(parent),
    type(type),
    fileName(fileName),
    ui(new Ui::ProcessFrame)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);
    setFixedSize(1600, 700);
    setWindowModality(Qt::ApplicationModal);

    QMenuBar* bar = new QMenuBar(ui->topFrame);
    bar->setFont(QFont("Times New Roman", 12));
    bar->setFixedSize(ui->topFrame->width(), ui->topFrame->height());

    QMenu *fileMenu = new QMenu("File", bar);
    fileMenu->setFont(QFont("Times New Roman", 12));
    bar->addMenu(fileMenu);

    QAction *selectBackAction = new QAction("select background", this);
    connect(selectBackAction, &QAction::triggered, this, [=]() {
        backgroundPath =
                QFileDialog::getOpenFileName(this, "Select Background", ".", "Image Files(*.jpg *.png)");
        if (backgroundPath == "") return;
        if (type == "Image") processImage(fileName);
        else if (type == "Video") {
            finished = false;
            ui->originLabel->setText("Processing......");
            ui->afterLabel->setText("Processing......");
            if (capture.isOpened()) capture.release();
            if (capture2.isOpened()) capture2.release();

            std::thread videoThread(&ProcessFrame::threadVideo, this);
            videoThread.detach();
        }
    });
    fileMenu->addAction(selectBackAction);

    QAction *saveToAction = new QAction("save to", this);
    connect(saveToAction, &QAction::triggered, this, [=]() {
        outputPath =
                QFileDialog::getSaveFileName(this, "Save to", ".", "Image Files(*.jpg)");
        if (outputPath == "") return;
        if (type == "Image") cv::imwrite(outputPath.toStdString(), after);
        else if (type == "Video") fs::copy_file(tmpFiles[0], outputPath.toStdString());
        else QMessageBox::information(this, "Notice", "Only image and video are supported.");
    });
    fileMenu->addAction(saveToAction);

    QAction *selectMatte = new QAction("select matte", this);
    connect(selectMatte, &QAction::triggered, this, [=]() {
        mattePath =
                QFileDialog::getOpenFileName(this, "Select matte", ".", "Image Files(*.jpg *.png)");
        if (mattePath == "") return;
        if (type == "Video") {
            finished = false;
            ui->originLabel->setText("Processing......");
            ui->afterLabel->setText("Processing......");
            if (capture.isOpened()) capture.release();
            if (capture2.isOpened()) capture2.release();

            std::thread videoThread(&ProcessFrame::threadVideo, this);
            videoThread.detach();
        }
        else QMessageBox::information(this, "Notice", "Only video is supported.");
    });
    fileMenu->addAction(selectMatte);

    if (type == "Image") processImage(fileName);
    else if (type == "Video") processVideo(fileName);
    else if (type == "Webcam") processWebcam(fileName);
}

ProcessFrame::~ProcessFrame()
{
    if (capture.isOpened()) capture.release();
    if (capture2.isOpened()) capture2.release();

    for (auto& tmp : tmpFiles) fs::remove(tmp);
    delete ui;
}

void ProcessFrame::showImage(QLabel *label, const cv::Mat &_image)
{
    cv::Mat image;
    cv::resize(_image, image, cv::Size(label->width(), label->height()));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qImage((uint8_t*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(qImage));
    label->show();
}

void ProcessFrame::processImage(const QString &fileName)
{
    clock_t t = handleImage(fileName.toStdString(),
                backgroundPath.toStdString(),
                origin, after);
    char buf[100];
    sprintf(buf, "%lld ms/frame", t);
    ui->speedLabel->setText(buf);

    showImage(ui->originLabel, origin);
    showImage(ui->afterLabel, after);
}

void ProcessFrame::threadVideo() {
    auto tmpPath = "./tmp_" + splitPath(fileName.toStdString(), '/').second;
    auto [t, miou] = handleVideo(fileName.toStdString(), mattePath.toStdString(),
                          backgroundPath.toStdString(), tmpPath);
    tmpFiles.push_back(tmpPath);

    capture.open(fileName.toStdString());
    capture2.open(tmpPath);

    tpf = t;
    finished = true;

    char buf[100];
    sprintf(buf, "%.2f ms/frame", t);
    ui->speedLabel->setText(buf);

    if (mattePath != "") {
        sprintf(buf, "%.2f%%", miou * 100);
        ui->miouLabel->setText(buf);
    }
    else ui->miouLabel->setText("None");
}

void ProcessFrame::processVideo(const QString &fileName)
{
    ui->originLabel->setAlignment(Qt::AlignCenter);
    ui->originLabel->setFont(QFont("Times New Roman", 14));
    ui->originLabel->setText("Processing......");

    ui->afterLabel->setAlignment(Qt::AlignCenter);
    ui->afterLabel->setFont(QFont("Times New Roman", 14));
    ui->afterLabel->setText("Processing......");

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [&]() {
        if (!finished) return;
        capture >> origin;
        capture2 >> after;
        if (!origin.empty() && !after.empty()) {
            showImage(ui->originLabel, origin);
            showImage(ui->afterLabel, after);
        }
        else {
            capture.release();
            capture2.release();

            capture.open(this->fileName.toStdString());
            capture2.open(this->tmpFiles[0]);
        }
    });

    capture.open(fileName.toStdString());
    double fps = capture.get(cv::CAP_PROP_FPS);
    capture.release();
    cout << 1000 / fps << endl;
    timer->start(1000 / fps);

    std::thread videoThread(&ProcessFrame::threadVideo, this);
    videoThread.detach();
}

void ProcessFrame::processWebcam(const QString &fileName)
{
    timer = new QTimer(this);
    capture.open(0);
    connect(timer, &QTimer::timeout, this, [&]() {
        clock_t t = handleWebcam(capture, origin, after, backgroundPath.toStdString());

        char buf[100];
        sprintf(buf, "%lld ms/frame", t);
        ui->speedLabel->setText(buf);
        showImage(ui->originLabel, origin);
        showImage(ui->afterLabel, after);
    });
    timer->start(20);
}
