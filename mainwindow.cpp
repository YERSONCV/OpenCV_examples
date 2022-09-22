#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <QIntValidator>
#include <QDebug>
#include <QThread>

using namespace std;
using namespace cv;

Mat dummy_1ch = Mat::zeros(Size(800, 600), CV_8UC1);
Mat dummy_3ch = Mat::zeros(Size(800, 600), CV_8UC3);
QImage dummy_img_1ch(dummy_1ch.data, dummy_1ch.cols, dummy_1ch.rows, QImage::Format_Grayscale8);
QImage dummy_img_3ch(dummy_3ch.data, dummy_3ch.cols, dummy_3ch.rows, QImage::Format_RGB888);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QIntValidator *portValidator = new QIntValidator(0, 255);
    QIntValidator *strideValidator = new QIntValidator(1, 50);

    ui->camPort->setValidator(portValidator);
    ui->dialDegree->setNotchesVisible(true);
    ui->strideEdit->setValidator(strideValidator);

    ui->SRC->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->SRC->width(), ui->SRC->height(), Qt::KeepAspectRatio));
    ui->ROI->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->ROI->width(), ui->ROI->height(), Qt::KeepAspectRatio));
    ui->imageFiltered->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageFiltered->width(), ui->imageFiltered->height(), Qt::KeepAspectRatio));
    ui->imageR->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageR->width(), ui->imageR->height(), Qt::KeepAspectRatio));
    ui->imageG->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageG->width(), ui->imageG->height(), Qt::KeepAspectRatio));
    ui->imageB->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageB->width(), ui->imageB->height(), Qt::KeepAspectRatio));
    ui->imageMerged->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageMerged->width(), ui->imageMerged->height(), Qt::KeepAspectRatio));
    ui->imageF->setPixmap(QPixmap::fromImage(dummy_img_1ch).scaled(ui->imageF->width(), ui->imageF->height(), Qt::KeepAspectRatio));

    connect(ui->PBTrecord, SIGNAL(clicked()), this, SLOT(record()));

    connect(ui->PBTrun, SIGNAL(clicked()), this, SLOT(runCamera()));
    connect(ui->dialDegree, &QDial::valueChanged, this, &MainWindow::dialValueChanged);
    connect(ui->spinDegree, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::spinValueChanged);
    connect(ui->PBTup, SIGNAL(clicked()), this, SLOT(pbtUpClicked()));
    connect(ui->PBTdown, SIGNAL(clicked()), this, SLOT(pbtDownClicked()));
    connect(ui->PBTleft, SIGNAL(clicked()), this, SLOT(pbtLeftClicked()));
    connect(ui->PBTright, SIGNAL(clicked()), this, SLOT(pbtRightClicked()));
    connect(ui->PBTplus, SIGNAL(clicked()), this, SLOT(pbtPlusClicked()));
    connect(ui->PBTminus, SIGNAL(clicked()), this, SLOT(pbtMinusClicked()));
    connect(ui->PBTstride, SIGNAL(clicked()), this, SLOT(pbtStrideClicked()));

    connect(ui->CBred, SIGNAL(clicked()), this, SLOT(cbRedClicked()));
    connect(ui->CBgreen, SIGNAL(clicked()), this, SLOT(cbGreenClicked()));
    connect(ui->CBblue, SIGNAL(clicked()), this, SLOT(cbBlueClicked()));
    connect(ui->CBsobelx, SIGNAL(clicked()), this, SLOT(cbSobelXClicked()));
    connect(ui->CBsobely, SIGNAL(clicked()), this, SLOT(cbSobelYClicked()));
    connect(ui->CBcorners, SIGNAL(clicked()), this, SLOT(cbCornersClicked()));

    connect(ui->CBcanny, SIGNAL(clicked()), this, SLOT(cbCannyClicked()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::runCamera(void){
    if(this->RUNNING==true){
        ui->SRC->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->SRC->width(), ui->SRC->height(), Qt::KeepAspectRatio));
        ui->ROI->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->ROI->width(), ui->ROI->height(), Qt::KeepAspectRatio));
        ui->imageFiltered->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageFiltered->width(), ui->imageFiltered->height(), Qt::KeepAspectRatio));
        ui->imageR->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageR->width(), ui->imageR->height(), Qt::KeepAspectRatio));
        ui->imageG->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageG->width(), ui->imageG->height(), Qt::KeepAspectRatio));
        ui->imageB->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageB->width(), ui->imageB->height(), Qt::KeepAspectRatio));
        ui->imageMerged->setPixmap(QPixmap::fromImage(dummy_img_3ch).scaled(ui->imageMerged->width(), ui->imageMerged->height(), Qt::KeepAspectRatio));
        ui->imageF->setPixmap(QPixmap::fromImage(dummy_img_1ch).scaled(ui->imageF->width(), ui->imageF->height(), Qt::KeepAspectRatio));

        ui->PBTrun->setText("Run");
    }else if(this->RUNNING==false){
        int port = ui->camPort->text().toInt();

        VideoCapture cap(port);

        if(!cap.isOpened()){
            QMessageBox::critical(this, "ERROR", "Please check your camera and port number.", QMessageBox::Ok);
            return;
        }

        this->camPort = port;
        ui->PBTrun->setText("Stop");

        this->RUNNING = true;

        Mat frame, gray, gray_, blur, EDGEX, EDGEY, EDGEC;
        Mat img_empty, img_red, img_green, img_blue, img_merged, img_filt;

        QImage src, roi, imgR, imgG, imgB, imgM, imgFil;

        Mat channels[3];
        Mat ROI, ROI_;

        double fps = cap.get(CAP_PROP_FPS);
        Size frameSize = Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
        int fourcc = VideoWriter::fourcc('X','V','I','D');
        vector<Mat> video;

        // Face detection
        CascadeClassifier face_classifier;
        face_classifier.load("/home/y3rsn/Dev/py/HaarCascade/Face_Detection/haarcascade_frontalface_default.xml");
        if(face_classifier.empty()){
            cout << "XML file not loaded!" << endl;
            this->FACE_CLASSIFIER = false;
        }
        this->FACE_CLASSIFIER = true;

        while(RUNNING==true){
            cap >> frame;

            // Recording video
            if(this->RECORDING_START){
                if(this->RECORDING_END==false){
                    Mat tmp;
                    frame.copyTo(tmp);
                    video.push_back(tmp);
                }else{
                    string filename = format("out%d.avi", this->VID_COUNT);
                    VideoWriter outputVideo(filename, fourcc, fps, frameSize, true);

                    for(size_t i = 0; i < video.size(); i++){
                        outputVideo.write(video.at(i));
                    }
                    outputVideo.release();
                    vector<Mat> video;

                    this->VID_COUNT++;
                    this->RECORDING_START = false;
                    this->RECORDING_END = false;
                }
            }

            img_empty = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);

            cvtColor(frame, frame, COLOR_BGR2RGB);
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
            split(frame, channels); // R, G, B

            this->width = frame.cols;
            this->height = frame.rows;
            this->CUR_X_RANGE_MAX = width - 100/this->MAG_CUR;
            this->CUR_Y_RANGE_MAX = height - 100/this->MAG_CUR;

            // ROI
            int original_width = frame.cols;
            int original_height = frame.rows;
            int ROI_x_left, ROI_x_right, ROI_y_top, ROI_y_bottom;
            int ROI_width = 100 / this->MAG_CUR;
            this->ROI_WIDTH = ROI_width;

            ROI_x_left = this->ROI_LEFT_TOP.y;
            ROI_y_top = this->ROI_LEFT_TOP.x;
            ROI_x_right = ROI_x_left + ROI_width;
            ROI_y_bottom = ROI_y_top + ROI_width;

            this->CUR_X_RANGE_MAX = original_width - ROI_width;
            this->CUR_Y_RANGE_MAX = original_height - ROI_width;

            ROI = frame(Range(ROI_x_left, ROI_x_right), Range(ROI_y_top, ROI_y_bottom));
            //qDebug() << this->MAG_CUR << ": " << ROI_width << ", " << ROI_x_left << ", " << ROI_x_right << ", " << ROI_y_top << ", " << ROI_y_bottom << endl;

            cv::resize(ROI, ROI_, Size(100, 100));

            // ROI rotation
            Mat rotation_matrix = getRotationMatrix2D(Point2f(ROI_.cols/2, ROI_.rows/2), (float)this->ROT_DEGREE, 1.0);
            warpAffine(ROI_, ROI_, rotation_matrix, Size(100, 100));

            roi = QImage(ROI_.data, ROI_.cols, ROI_.rows, QImage::Format_RGB888);
            ui->ROI->setPixmap(QPixmap::fromImage(roi).scaled(ui->ROI->width(), ui->ROI->height(), Qt::KeepAspectRatio));

            // R-channel
            vector<Mat> mergedR;
            mergedR.push_back(channels[0]);
            mergedR.push_back(img_empty);
            mergedR.push_back(img_empty);
            merge(mergedR, img_red);
            imgR = QImage(img_red.data, img_red.cols, img_red.rows, QImage::Format_RGB888);
            ui->imageR->setPixmap(QPixmap::fromImage(imgR).scaled(ui->imageR->width(), ui->imageR->height(), Qt::KeepAspectRatio));

            // G-channel
            vector<Mat> mergedG;
            mergedG.push_back(img_empty);
            mergedG.push_back(channels[1]);
            mergedG.push_back(img_empty);
            merge(mergedG, img_green);
            imgG = QImage(img_green.data, img_green.cols, img_green.rows, QImage::Format_RGB888);
            ui->imageG->setPixmap(QPixmap::fromImage(imgG).scaled(ui->imageG->width(), ui->imageG->height(), Qt::KeepAspectRatio));

            // B-channel
            vector<Mat> mergedB;
            mergedB.push_back(img_empty);
            mergedB.push_back(img_empty);
            mergedB.push_back(channels[2]);
            merge(mergedB, img_blue);
            imgB = QImage(img_blue.data, img_blue.cols, img_blue.rows, QImage::Format_RGB888);
            ui->imageB->setPixmap(QPixmap::fromImage(imgB).scaled(ui->imageB->width(), ui->imageB->height(), Qt::KeepAspectRatio));

            // merged image
            vector<Mat> mergedM;
            if(this->R_CHANNEL){
                mergedM.push_back(channels[0]);
            }else{
                mergedM.push_back(img_empty);
            }

            if(this->G_CHANNEL){
                mergedM.push_back(channels[1]);
            }else{
                mergedM.push_back(img_empty);
            }

            if(this->B_CHANNEL){
                mergedM.push_back(channels[2]);
            }else{
                mergedM.push_back(img_empty);
            }
            merge(mergedM, img_merged);
            imgM = QImage(img_merged.data, img_merged.cols, img_merged.rows, QImage::Format_RGB888);
            ui->imageMerged->setPixmap(QPixmap::fromImage(imgM).scaled(ui->imageMerged->width(), ui->imageMerged->height(), Qt::KeepAspectRatio));

            Mat sobelx, sobely, canny, result;
            vector<Mat> sx, sy, ca, ha;

            if(this->SOBEL_X){
                Sobel(gray, sobelx, CV_32F, 1, 0);
                sobelx.convertTo(sobelx, CV_8UC1);
            }else{
                sobelx = img_empty;
            }
            sx.push_back(img_empty);
            sx.push_back(sobelx);
            sx.push_back(sobelx);
            merge(sx, EDGEX);

            if(this->SOBEL_Y){
                Sobel(gray, sobely, CV_32F, 0, 1);
                sobely.convertTo(sobely, CV_8UC1);
            }else{
                sobely = img_empty;
            }
            sy.push_back(sobely);
            sy.push_back(img_empty);
            sy.push_back(sobely);
            merge(sy, EDGEY);

            if(this->CANNY){
                Canny(gray, canny, 100, 200);
            }else{
                canny = img_empty;
            }
            ca.push_back(canny);
            ca.push_back(canny);
            ca.push_back(canny);
            merge(ca, EDGEC);

            addWeighted(EDGEX, 1, EDGEY, 1, 0, result);
            addWeighted(EDGEC, 1, result, 1, 0, result);

            if(this->CORNERS){
                Mat mask;
                vector<Point2f> corners;
                goodFeaturesToTrack(gray, corners, 20, 0.01, 20., mask, 3, false, 0.04);
                for(size_t i = 0; i < corners.size(); i++){
                    cv::circle(result, corners[i], 10, Scalar(0, 255, 0), -1);
                }
            }
            imgFil = QImage(result.data, result.cols, result.rows, QImage::Format_RGB888);
            ui->imageFiltered->setPixmap(QPixmap::fromImage(imgFil).scaled(ui->imageFiltered->width(), ui->imageFiltered->height(), Qt::KeepAspectRatio));
            Mat gray_;
            //gray.copyTo(gray_);
            //fourier_transform(gray_);

            // DFT
            Mat fimage, dftA, dftA2[2], magImage, magF;
            QImage imgF;

            gray.convertTo(fimage, CV_32F);

            dft(fimage, dftA, DFT_COMPLEX_OUTPUT);
            split(dftA, dftA2);
            //qDebug() << dftA.cols << ", " << dftA2[0].cols << endl;
            // Spectrum
            magnitude(dftA2[0], dftA2[1], magF);
            //qDebug() << magF.cols << endl;
            magF += Scalar(1);
            log(magF, magF);

            normalize(magF, magImage, 0, 255, NORM_MINMAX, CV_8U);

            /*
            // Phase angle
            Mat angleF;
            phase(dftA2[0], dftA2[1], angleF);

            Mat angleImage;
            normalize(angleF, angleImage, 0, 255, NORM_MINMAX, CV_8U);
            */

            imgF = QImage(magImage.data, magImage.cols, magImage.rows, QImage::Format_Grayscale8);
            ui->imageF->setPixmap(QPixmap::fromImage(imgF).scaled(ui->imageF->width(), ui->imageF->height(), Qt::KeepAspectRatio));

            // Draw ROI on SRC
            rectangle(frame, Point(ROI_y_top, ROI_x_left), Point(ROI_y_bottom, ROI_x_right), Scalar(0, 255, 0), 2, LINE_8, 0);

            // Draw detected faces
            if(this->FACE_CLASSIFIER){
                gray.copyTo(gray_);
                vector<Rect> faces;
                face_classifier.detectMultiScale(gray_, faces, 1.1, 3);

                for(size_t i = 0; i < faces.size(); i++){
                    rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 3);
                }
            }

            src = QImage(frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
            ui->SRC->setPixmap(QPixmap::fromImage(src).scaled(ui->SRC->width(), ui->SRC->height(), Qt::KeepAspectRatio));

            waitKey(1);
        }
        cap.release();
    }
}

void MainWindow::pbtStrideClicked(void){
    this->STRIDE = ui->strideEdit->text().toInt();
}

void MainWindow::dialValueChanged(void){
    int val = ui->dialDegree->value();
    this->ROT_DEGREE = val;
    ui->spinDegree->setValue(val);
}

void MainWindow::spinValueChanged(void){
    int val = ui->spinDegree->value();
    this->ROT_DEGREE = val;
    ui->dialDegree->setValue(val);
}

void MainWindow::pbtUpClicked(void){
    int x = this->ROI_LEFT_TOP.x;
    int y = this->ROI_LEFT_TOP.y;

    if(y-this->STRIDE < 0){
       return;
    }else{
        y-=this->STRIDE;
    }
    this->ROI_LEFT_TOP = Point(x, y);
}

void MainWindow::pbtDownClicked(void){
    int x = this->ROI_LEFT_TOP.x;
    int y = this->ROI_LEFT_TOP.y;

    if(y+this->STRIDE > this->height-this->ROI_WIDTH-1){
        return;
    }else{
        y+=this->STRIDE;
    }
    this->ROI_LEFT_TOP = Point(x, y);
}

void MainWindow::pbtLeftClicked(void){
    int x = this->ROI_LEFT_TOP.x;
    int y = this->ROI_LEFT_TOP.y;

    if(x-this->STRIDE < 0){
        return;
    }else{
        x-=this->STRIDE;
    }
    this->ROI_LEFT_TOP = Point(x, y);
}

void MainWindow::pbtRightClicked(void){
    int x = this->ROI_LEFT_TOP.x;
    int y = this->ROI_LEFT_TOP.y;

    if(x+this->STRIDE > this->width-this->ROI_WIDTH-1){
        return;
    }else{
        x+=STRIDE;
    }
    this->ROI_LEFT_TOP = Point(x, y);
}

void MainWindow::pbtPlusClicked(void){
    if(this->MAG_CUR != this->MAG_MAX) this->MAG_CUR++;

}

void MainWindow::pbtMinusClicked(void){
    if(this->MAG_CUR != this->MAG_MIN) this->MAG_CUR--;
}

void MainWindow::cbRedClicked(void){
    if(ui->CBred->isChecked()){
        this->R_CHANNEL = true;
    }else{
        this->R_CHANNEL = false;
    }
}

void MainWindow::cbGreenClicked(void){
    if(ui->CBgreen->isChecked()){
        this->G_CHANNEL = true;
    }else{
        this->G_CHANNEL = false;
    }
}

void MainWindow::cbBlueClicked(void){
    if(ui->CBblue->isChecked()){
        this->B_CHANNEL = true;
    }else{
        this->B_CHANNEL = false;
    }
}

void MainWindow::cbSobelXClicked(void){
    if(ui->CBsobelx->isChecked()){
        this->SOBEL_X = true;
    }else{
        this->SOBEL_X = false;
    }
}

void MainWindow::cbSobelYClicked(void){
    if(ui->CBsobely->isChecked()){
        this->SOBEL_Y = true;
    }else{
        this->SOBEL_Y = false;
    }
}

void MainWindow::cbCornersClicked(void){
    if(ui->CBcorners->isChecked()){
        this->CORNERS = true;
    }else{
        this->CORNERS = false;
    }
}

void MainWindow::cbCannyClicked(void){
    if(ui->CBcanny->isChecked()){
        this->CANNY = true;
    }else{
        this->CANNY = false;
    }
}

void MainWindow::record(void){
    if(this->RECORDING_START){
        ui->PBTrecord->setText("RECORD");
        this->RECORDING_END=true;
    }else{
        ui->PBTrecord->setText("STOP");
        this->RECORDING_START=true;
    }
}
