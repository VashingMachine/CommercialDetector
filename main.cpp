#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <MovedAverage.h>
#include <CommercialClock.h>
#include <iostream>

#define SCALE 300

using namespace cv;
using namespace dnn;

class SimpleCommericalsDetector {
protected:
    Net net;
    MovedAverage movedAverage;

    bool openSource(cv::VideoCapture &cap, std::string source) {
        return source == "0" ? cap.open(0) : cap.open(source);
    }

    void registerWindows() {
        namedWindow("TV", WINDOW_AUTOSIZE);
    }

    void prepareBlob(Mat &source, Mat &targetBlob) {
        Mat gray;
        cvtColor(source, gray, COLOR_RGB2GRAY);
        Scalar means, deviations;
        meanStdDev(gray, means, deviations);
        float targetScale = 0.5f / deviations[0];
        targetBlob = blobFromImage(source, targetScale, Size(SCALE, (int)(720. / 1280 * SCALE)), 0.5);
    }

public:
    explicit SimpleCommericalsDetector(std::string model): movedAverage(50) {
        net = readNetFromONNX(model);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        registerWindows();
    }

    void run(std::string video_source) {
        VideoCapture camera;
        openSource(camera, video_source);
        int frameRate = camera.get(CAP_PROP_FPS);
        CommercialClock clock(frameRate);
        int frameIdx = 0;
        while (camera.isOpened()) {
            Mat frame, blob;
            camera >> frame;
            if (frame.empty()) {
                clock.finish(frameIdx);
                std::cout << clock.statistics();
                break;
//                camera.set(CAP_PROP_POS_FRAMES, 0);
            } else {
                prepareBlob(frame, blob);
                net.setInput(blob);
                Mat result = net.forward();
                movedAverage.addProb(result.at<float>(0, 0));
                bool isCommercial = movedAverage.getAverage() > 0.5;
                clock.registerType(isCommercial, frameIdx);
                if(isCommercial) {
                    putText(frame, //target image
                                "Commercial", //text
                                cv::Point(10, frame.rows - 20), //top-left position
                                cv::FONT_HERSHEY_DUPLEX,
                                1.0,
                                CV_RGB(118, 185, 0), //font color
                                2);
                } else {
                    putText(frame, //target image
                            "Movie", //text
                            cv::Point(10, frame.rows - 20), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(118, 185, 0), //font color
                            2);
                }
                imshow("TV", frame);
            }
            frameIdx++;
            int k = waitKey(15);
            if(k == 'k' || k == 'K') {
                break;
            }
        }
        destroyAllWindows();
        camera.release();
    }
};

int main(int argc, char** argv)
{
    SimpleCommericalsDetector scd("/home/ktoztam/CLionProjects/CommercialDetector/classifier/polsat_detector.onnx");
    scd.run("/home/ktoztam/CLionProjects/CommercialDetector/classifier/data/MaszDosc.mp4");
}