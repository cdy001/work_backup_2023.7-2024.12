#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(){
    string img_path = "/data/cdy/cpp_project/R01C04L1.bmp";
    cout << "image_path: " << img_path << endl;

    // 读取图像
    Mat image = imread(img_path, -1);

    // 检查图像是否成功读取
    if (image.empty()) {
        cerr << "Could not open or find the image." << endl;
        return -1;
    }

    // 二值化
    Mat image_thr;
    threshold(InputArray(image), OutputArray(image_thr), 250, 255, THRESH_BINARY_INV);

    // 形态学
    Mat morph_kernel = getStructuringElement(MORPH_RECT, Size2d(30, 30));
    Mat image_morph;
    morphologyEx(InputArray(image_thr), OutputArray(image_morph), MORPH_CLOSE, morph_kernel);
    
    // 保存图像
    imwrite("../result_img.bmp", image_morph);

    return 0;
}