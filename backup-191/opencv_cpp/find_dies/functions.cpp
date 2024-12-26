#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "nlohmann/json.hpp"


auto time_now()
{
    auto dura = std::chrono::system_clock::now().time_since_epoch();
    auto result = std::chrono::duration_cast<std::chrono::milliseconds>(dura).count();

    return result;
}


struct Die {
    int x_center, y_center, x_min, y_min, x_max, y_max;
    int die_flag_1, die_flag_2, die_flag_3, die_flag_4;
};


nlohmann::json read_json_file(std::string file_path) {
    // 打开 JSON 文件
    std::ifstream j_file(file_path);
    if (!j_file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
    }
    // 解析 JSON 文件
    nlohmann::json json_dict;
    j_file >> json_dict;

    return json_dict;
}


std::tuple<cv::Mat, std::vector<std::vector<cv::Point>>> retrive_contours(cv::String img_path, std::unordered_map<std::string, int> args){
    std::cout << "image_path: " << img_path << std::endl;
    int thresh = args["thresh"];
    int binary_type = args["binary_type"];
    int morph_type = args["morph_type"];
    int morph_size_x = args["morph_size_x"];
    int morph_size_y = args["morph_size_y"];

    auto time_1 = time_now();

    // 读取图像
    cv::Mat image = cv::imread(img_path, -1);

    auto time_2 = time_now();
    std::cout << "time read img: " << (time_2 - time_1) * 0.001 << std::endl;

    // 二值化
    cv::Mat image_thr;
    cv::threshold(cv::InputArray(image), cv::OutputArray(image_thr), thresh, 255, binary_type);

    auto time_3 = time_now();
    std::cout << "time threshold: " << (time_3 - time_2) * 0.001 << std::endl;

    // 形态学
    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size2d(morph_size_x, morph_size_y));
    cv::Mat image_morph;
    cv::morphologyEx(cv::InputArray(image_thr), cv::OutputArray(image_morph), morph_type, morph_kernel);

    auto time_4 = time_now();
    std::cout << "time morphlogy: " << (time_4 - time_3) * 0.001 << std::endl;

    // retrive contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(cv::InputArray(image_morph), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return std::make_tuple(image_morph, contours);
}


std::tuple<std::vector<Die>, std::vector<Die>> contours2rects(
    const std::vector<std::vector<cv::Point>>& contours, std::unordered_map<std::string, int> args
    ) {
        int die_width = args["die_width"], die_height = args["die_height"];
        int img_width = args["img_width"], img_height = args["img_height"];
        std::vector<Die> dies, dies_long_vertical, dies_long_horizon, dies_long_big;
        int spec_w = die_width * 0.9, spec_h = die_height * 0.8, spec_twins_w = die_width * 1.2, spec_twins_h = die_height * 1.2;
        int die_flag_1=0, die_flag_2=0, die_flag_3=0, die_flag_4=0;
        int die_nums = 0;
        double die_area = die_width * die_height;

        // 处理每个轮廓
        for (const auto& contourPoint : contours) {
            double contour_area = cv::contourArea(contourPoint);
            
            // 去掉轮廓内面积较小的情况
            if (contour_area < 0.8 * die_area) {
                continue;
            }

            // 获取包围框
            cv::Rect bounding_rect = cv::boundingRect(contourPoint);
            int x = bounding_rect.x;
            int y = bounding_rect.y;
            int w = bounding_rect.width;
            int h = bounding_rect.height;

            // 去除边缘切得不完整的芯粒
            if (x < 2 || y < 2 || x + w > img_width - 2 || y + h > img_height - 2) {
                if ((2.5 * spec_w > w && w > spec_w && 2.5 * spec_h > h && h > spec_h) || w > 5000 || h > 5000) {
                    continue;
                }
            }

            // 水平和垂直多胞
            if (spec_twins_w < w && w < 5 * spec_twins_w && spec_twins_h < h && h < 5 * spec_twins_h) {
                // 如果是较大的多胞
                int die_width_count = round(abs(w) / die_width);
                int die_height_count = round(abs(h) / die_height);
                int new_die_width = int(w / die_width_count);
                int new_die_height = int(h / die_height_count);

                for (int j = 0; j < die_height_count; j++) {
                    int new_die_left_y = y + j * new_die_height;
                    int new_die_right_y = new_die_left_y + new_die_height;
                    for (int k = 0; k < die_width_count; k++) {
                        int new_die_left_x = x + k * new_die_width;
                        int new_die_right_x = new_die_left_x + new_die_width;
                        dies_long_big.push_back(
                            {
                                int((new_die_left_x + new_die_right_x) / 2),
                                int((new_die_right_y + new_die_left_y) / 2),
                                new_die_left_x,
                                new_die_left_y,
                                new_die_right_x,
                                new_die_right_y,
                                2,
                                die_flag_2,
                                die_flag_3,
                                die_flag_4,
                            }
                        );
                    }
                }
            }

            // 垂直多胞
            else if (2 * spec_w > w && w > spec_w && 5 * spec_twins_h > h && h > spec_twins_h){
                int die_count = round(abs(h) / die_height);
                int new_die_height = int(abs(h) / die_count);
                for (int j = 0; j < die_count; j++) {
                    int new_die_left_y = y + j * new_die_height;
                    int new_die_right_y = new_die_left_y + new_die_height;
                    dies_long_vertical.push_back(
                        {
                            int(x + w / 2),
                            int((new_die_left_y + new_die_right_y) / 2),
                            x,
                            new_die_left_y,
                            x + w,
                            new_die_right_y,
                            2,
                            die_flag_2,
                            die_flag_3,
                            die_flag_4,
                        }
                    );
                }
            }

            // 水平多胞
            else if (2 * spec_h > h && h > spec_h && 5 * spec_twins_w > w && w > spec_twins_w){
                int die_count = round(abs(w) / die_width);
                int new_die_width = int(abs(w) / die_count);
                for (int j = 0; j < die_count; j++) {
                    int new_die_left_x = x + j * new_die_width;
                    int new_die_right_x = new_die_left_x + new_die_width;
                    dies_long_horizon.push_back(
                        {
                            int((new_die_left_x + new_die_right_x) / 2),
                            int(y + h / 2),
                            new_die_left_x,
                            y,
                            new_die_right_x,
                            y + h,
                            2, 
                            die_flag_2,
                            die_flag_3,
                            die_flag_4,
                        }
                    );
                }
            }

            // 正常的die
            else if (1.5 * spec_w > w && w > spec_w && 1.5 * spec_h > h && h > spec_h) {
                dies.push_back({x + int(w / 2), y + int(h / 2), x, y, x + w, y + h, die_flag_1, die_flag_2, die_flag_3, die_flag_4});
            }
        }
        dies_long_big.insert(dies_long_big.end(), std::make_move_iterator(dies_long_vertical.begin()), std::make_move_iterator(dies_long_vertical.end()));
        dies_long_big.insert(dies_long_big.end(), std::make_move_iterator(dies_long_horizon.begin()), std::make_move_iterator(dies_long_horizon.end()));
        return std::make_tuple(dies, dies_long_big);
        }
