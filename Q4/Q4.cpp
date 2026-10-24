#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <intrin.h>
#include <functional>

using namespace cv;
using namespace std;
using namespace chrono;

constexpr int MOTION_THRESHOLD = 30;
constexpr char QUIT_KEY = 'q';

enum DetectionMode { SERIAL, PARALLEL };

struct DetectionResult {
    uint64_t duration = 0;
    float speedup = 0.0f;
};

bool fetchGrayFrame(VideoCapture& cap, Mat& frame) {
    cap >> frame;
    if (frame.empty()) return false;
    cvtColor(frame, frame, COLOR_BGR2GRAY);
    return true;
}

void processMotion(VideoCapture& video,
                   DetectionMode mode,
                   DetectionResult& result,
                   const function<void(Mat&, Mat&, Mat&)>& motionLogic) {
    Mat prevFrame, currFrame, motionFrame;
    result.duration = 0;

    if (!fetchGrayFrame(video, prevFrame)) {
        cerr << "Error: Cannot load the first frame." << endl;
        return;
    }

    const char* windowTitle = (mode == SERIAL) ? "Serial Motion Detected" : "Parallel Motion Detected";
    namedWindow("Video Feed", WINDOW_NORMAL);
    namedWindow(windowTitle, WINDOW_NORMAL);

    while (fetchGrayFrame(video, currFrame)) {
        motionFrame = Mat::zeros(currFrame.size(), CV_8UC1);

        auto startTime = high_resolution_clock::now();
        motionLogic(prevFrame, currFrame, motionFrame);
        auto endTime = high_resolution_clock::now();

        result.duration += duration_cast<microseconds>(endTime - startTime).count();

        imshow("Video Feed", currFrame);
        imshow(windowTitle, motionFrame);

        prevFrame = currFrame.clone();

        if (waitKey(30) == QUIT_KEY) break;
    }

    video.release();
    destroyAllWindows();
}

void serialMotionLogic(Mat& prevFrame, Mat& currFrame, Mat& motionFrame) {
    int rows = currFrame.rows, cols = currFrame.cols;
    for (int i = 0; i < rows; i++) {
        const uchar* prevRow = prevFrame.ptr<uchar>(i);
        const uchar* currRow = currFrame.ptr<uchar>(i);
        uchar* motionRow = motionFrame.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            motionRow[j] = abs(currRow[j] - prevRow[j]) > MOTION_THRESHOLD ? 255 : 0;
        }
    }
}

void parallelMotionLogic(Mat& prevFrame, Mat& currFrame, Mat& motionFrame) {
    int rows = currFrame.rows, cols = currFrame.cols;
    __m128i threshold = _mm_set1_epi8(MOTION_THRESHOLD);

    for (int i = 0; i < rows; i++) {
        const uchar* prevRow = prevFrame.ptr<uchar>(i);
        const uchar* currRow = currFrame.ptr<uchar>(i);
        uchar* motionRow = motionFrame.ptr<uchar>(i);

        for (int j = 0; j < cols; j += 16) {
            __m128i prevPixels = _mm_load_si128((__m128i*)&prevRow[j]);
            __m128i currPixels = _mm_load_si128((__m128i*)&currRow[j]);
            __m128i diff = _mm_abs_epi8(_mm_sub_epi8(currPixels, prevPixels));
            __m128i motionPixels = _mm_cmpgt_epi8(diff, threshold);
            _mm_storeu_si128((__m128i*)&motionRow[j], motionPixels);
        }
    }
}

void runDetection(const string& videoPath, DetectionMode mode, DetectionResult& result) {
    VideoCapture video(videoPath);
    if (!video.isOpened()) {
        cerr << "Error: Unable to open video file." << endl;
        return;
    }


    auto motionLogic = (mode == SERIAL) ? serialMotionLogic : parallelMotionLogic;
    processMotion(video, mode, result, motionLogic);
}

void displayResults(const DetectionResult& serialResult, const DetectionResult& parallelResult) {
    cout << "Serial Run Time    = " << serialResult.duration << " microseconds" << endl;
    cout << "Parallel Run Time  = " << parallelResult.duration << " microseconds" << endl;
    if (parallelResult.duration > 0) {
        cout << "Speedup            = " << static_cast<float>(serialResult.duration) / parallelResult.duration << endl;
    }
}

int main() {
    string videoPath = "F:/Program Files (x86)/MMD.Soor/University/Term 7/PP/CAs/CA1/PP_CA1/assets/Q4/Q4.mp4";

    DetectionResult serialResult, parallelResult;

    runDetection(videoPath, SERIAL, serialResult);
    runDetection(videoPath, PARALLEL, parallelResult);

    displayResults(serialResult, parallelResult);

    return 0;
}
