#include <iostream>
#include "opencv2/opencv.hpp"
#include "nlohmann/json.hpp"

#include "functions.h"



int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path, save_path, config_path>" << std::endl;
        return -1;
    }

    cv::String img_path = argv[1];  //argv[0]是程序名
    cv::String save_path = argv[2];
    std::string json_file_path = argv[3];  //配置文件
    
    nlohmann::json json_dict = read_json_file(json_file_path);
    std::unordered_map<std::string, int> args;
    for (const auto& item: json_dict.items()) {
        args[item.key()] = int(item.value());
    }

    cv::Mat image_save = cv::imread(img_path, 1);

    auto [_, contours] = retrive_contours(img_path=img_path, args=args);
    auto [dies, dies_long_big] = contours2rects(contours=contours, args=args);
    for (const Die die: dies){
        // cout << die.x_min << " " << die.y_min << " " << die.x_max << " " << die.y_max << endl;
        cv::Point pt1(die.x_min - args["margin_x"], die.y_min - args["margin_y"]);
        cv::Point pt2(die.x_max + args["margin_x"], die.y_max + args["margin_y"]);
        cv::rectangle(image_save, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }
    for (const Die die: dies_long_big){
        // cout << die.x_min << " " << die.y_min << " " << die.x_max << " " << die.y_max << endl;
        cv::Point pt1(die.x_min - args["margin_x"], die.y_min - args["margin_y"]);
        cv::Point pt2(die.x_max + args["margin_x"], die.y_max + args["margin_y"]);
        cv::rectangle(image_save, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }

    // 保存图像
    cv::imwrite(save_path, image_save);

    return 0;
}