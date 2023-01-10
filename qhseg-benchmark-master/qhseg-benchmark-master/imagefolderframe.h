#ifndef IMAGEFOLDERFRAME_H
#define IMAGEFOLDERFRAME_H

#include <QFrame>

namespace Ui {
class ImageFolderFrame;
}

class ImageFolderFrame : public QFrame
{
    Q_OBJECT

public:
    explicit ImageFolderFrame(QWidget *parent = nullptr);
    ~ImageFolderFrame();

    void processFolder();
    void folderThread();

private:
    Ui::ImageFolderFrame *ui;
    QString imageFolderPath;
    QString matteFolderPath;
    QString outputFolderPath;
    QString backgroundPath;
};

#endif // IMAGEFOLDERFRAME_H
