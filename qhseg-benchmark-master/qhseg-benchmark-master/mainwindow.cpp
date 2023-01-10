#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "processframe.h"
#include "processing.h"
#include "imagefolderframe.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(800, 600);

    QString fileName;
    while (fileName == "") {
        fileName = QFileDialog::getOpenFileName(this,  "Choose Human Segmentation Model",  ".");
        if (fileName == "")
            QMessageBox::warning(this, "Warning", "Please choose one model!");
    }
    setPredictor(fileName.toStdString());

    connect(ui->imagePushButton, &QPushButton::clicked, this, &MainWindow::chooseImage);
    connect(ui->videoPushButton, &QPushButton::clicked, this, &MainWindow::chooseVideo);
    connect(ui->webcamPushButton, &QPushButton::clicked, this, &MainWindow::chooseWebcam);
    connect(ui->folderPushButton, &QPushButton::clicked, this, &MainWindow::chooseFolder);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::chooseImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Choose Image", ".", "Image Files(*.jpg *.png)");
    if (fileName != "") {
        auto frame = new ProcessFrame(nullptr, "Image", fileName);
        frame->show();
    }
}

void MainWindow::chooseVideo()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Choose Video", ".", "Video Files(*.avi *.mp4)");
    if (fileName != "") {
        auto frame = new ProcessFrame(nullptr, "Video", fileName);
        frame->show();
    }
}

void MainWindow::chooseWebcam()
{
    auto frame = new ProcessFrame(nullptr, "Webcam", "");
    frame->show();
}

void MainWindow::chooseFolder()
{
    auto frame = new ImageFolderFrame(nullptr);
    frame->show();
}

