#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;

int main() {
    CascadeClassifier faceCascade, eyesCascade, smileCascade;

    if (!faceCascade.load("C:/Users/User/Desktop/haarcascade_frontalface_alt.xml") ||
        !eyesCascade.load("C:/Users/User/Desktop/haarcascade_eye_tree_eyeglasses.xml") ||
        !smileCascade.load("C:/Users/User/Desktop/haarcascade_smile.xml")) {
        std::cout << "Error" << std::endl;
        return -1;
    }

    VideoCapture cap("C:/Users/User/Downloads/video.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error!" << std::endl;
        return -1;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter videoWriter("output.mp4", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20, Size(frame_width, frame_height));

    while (true) {

        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video" << std::endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        std::vector<Rect> faces, eyes, smiles;
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(255, 170, 15), 2);

            eyesCascade.detectMultiScale(gray(face), eyes, 3, 2, 0, Size(5, 5));
            for (const auto& eye : eyes) {
                Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);
                circle(frame(face), center, radius, Scalar(255, 0, 255), 2);
            }

            smileCascade.detectMultiScale(gray(face), smiles, 1.565, 30, 0, Size(30, 30));
            for (const auto& smile : smiles) {
                rectangle(frame(face), smile, Scalar(255, 255, 0), 2);
            }

            blur(frame(face), frame(face), Size(3, 3));
        }

        videoWriter.write(frame);
        imshow("Face Detection", frame);

        if (waitKey(25) == 'q') {
            break;
        }
    }
    videoWriter.release();
    destroyAllWindows();
    return 0;
}