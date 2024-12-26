#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

auto time_now()
{
    auto dura = chrono::system_clock::now().time_since_epoch();
    auto result = chrono::duration_cast<chrono::milliseconds>(dura).count();

    return result;
}

int main()
{
    string img_path = "/data/cdy/cpp_project/R01C04L1.bmp";
    cout << "image_path: " << img_path << endl;

    auto time_1 = time_now();

    // 读取图像
    Mat image = imread(img_path, -1);

    // 检查图像是否成功读取
    if (image.empty()) {
        cerr << "Could not open or find the image." << endl;
        return -1;
    }

    auto time_2 = time_now();
    cout << "time read img: " << (time_2 - time_1) * 0.001 << endl;

    // 二值化
    Mat image_thr;
    threshold(InputArray(image), OutputArray(image_thr), 250, 255, THRESH_BINARY_INV);

    auto time_3 = time_now();
    cout << "time threshold: " << (time_3 - time_2) * 0.001 << endl;

    // 形态学
    Mat morph_kernel = getStructuringElement(MORPH_RECT, Size2d(30, 30));
    Mat image_morph;
    morphologyEx(InputArray(image_thr), OutputArray(image_morph), MORPH_CLOSE, morph_kernel);

    auto time_4 = time_now();
    cout << "time morphlogy: " << (time_4 - time_3) * 0.001 << endl;
    
    // 保存图像
    imwrite("../result_img.bmp", image_morph);

    return 0;
}