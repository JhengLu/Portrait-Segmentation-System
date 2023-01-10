#include "imagefolderframe.h"
#include "ui_imagefolderframe.h"
#include "processing.h"

#include <QFileDialog>

ImageFolderFrame::ImageFolderFrame(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::ImageFolderFrame)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);
    setFixedSize(800, 600);
    setWindowModality(Qt::ApplicationModal);

    connect(ui->imagePushButton, &QPushButton::clicked, this, [=]() {
        imageFolderPath = QFileDialog::getExistingDirectory(this, "Choose Image Folder", ".");
        ui->imageLabel->setText(imageFolderPath);
    });
    connect(ui->mattePushButton, &QPushButton::clicked, this, [=]() {
        matteFolderPath = QFileDialog::getExistingDirectory(this, "Choose Matte Folder", ".");
        ui->matteLabel->setText(matteFolderPath);
    });
    connect(ui->outputPushButton, &QPushButton::clicked, this, [=]() {
        outputFolderPath = QFileDialog::getExistingDirectory(this, "Choose Output Folder", ".");
        ui->outputLabel->setText(outputFolderPath);
    });
    connect(ui->backgroundPushButton, &QPushButton::clicked, this, [=]() {
        backgroundPath = QFileDialog::getOpenFileName(this, "Choose Background", ".", "Image Files(*.jpg *.png)");
        ui->backgroundLabel->setText(backgroundPath);
    });
    connect(ui->okPushButton, &QPushButton::clicked, this, &ImageFolderFrame::processFolder);
}

ImageFolderFrame::~ImageFolderFrame()
{
    delete ui;
}

void ImageFolderFrame::processFolder()
{
    auto [t, miou] = handleImageFolder(imageFolderPath.toStdString(),
                      matteFolderPath.toStdString(),
                      backgroundPath.toStdString(),
                      outputFolderPath.toStdString());
    char buf[100];
    sprintf(buf, "%.2f ms/frame", t);
    ui->speedLabel->setText(buf);

    if (miou != -1) {
        sprintf(buf, "%.2f%%", miou * 100);
        ui->miouLabel->setText(buf);
    }
    else ui->miouLabel->setText("None");
}
