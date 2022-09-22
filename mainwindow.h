#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QString>
#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    bool RUNNING            = false;
    bool R_CHANNEL          = false;
    bool G_CHANNEL          = false;
    bool B_CHANNEL          = false;
    bool SOBEL_X            = false;
    bool SOBEL_Y            = false;
    bool CORNERS            = false;
    bool CANNY              = false;
    bool RECORDING_START    = false;
    bool RECORDING_END      = false;
    bool FACE_CLASSIFIER    = false;

    int VID_COUNT = 0;

    int MAG_MAX             = 5;
    int MAG_MIN             = 1;
    int MAG_CUR             = 1;
    int CUR_X_RANGE_MAX;
    int CUR_Y_RANGE_MAX;
    int STRIDE              = 1;
    int ROI_WIDTH           = 100;

    int ROT_DEGREE          = 0;
    int width;
    int height;

    int camPort             = 0;
    Point ROI_LEFT_TOP      = Point(0, 0);

public slots:
    // Main capture image pannel
    void runCamera(void);

    // ROI control pannel
    void dialValueChanged(void);
    void spinValueChanged(void);
    void pbtUpClicked(void);
    void pbtDownClicked(void);
    void pbtLeftClicked(void);
    void pbtRightClicked(void);
    void pbtPlusClicked(void);
    void pbtMinusClicked(void);
    void pbtStrideClicked(void);

    // Check box options
    void cbRedClicked(void);
    void cbGreenClicked(void);
    void cbBlueClicked(void);

    void cbSobelXClicked(void);
    void cbSobelYClicked(void);
    void cbCornersClicked(void);
    void cbCannyClicked(void);

    // record option
    void record(void);

};
#endif // MAINWINDOW_H
