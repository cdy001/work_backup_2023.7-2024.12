#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    // 读取图像
    Mat image = imread("/data/chendeyang/code/torch-tensorrt/IMAGE1_0043.bmp");

    // 检查图像是否成功读取
    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // 输出图像的信息
    std::cout << "Width: " << image.cols << std::endl;
    std::cout << "Height: " << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    std::cout << "Depth: " << image.depth() << std::endl;
    std::cout << "Type: " << image.type() << std::endl;

    // // 显示图像
    // cv::imshow("Display Image", image);
    // cv::waitKey(0); // 等待按键事件

    return 0;
}