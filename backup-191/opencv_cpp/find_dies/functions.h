#ifndef CONTOURS_H
#define CONTOURS_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "nlohmann/json.hpp"


struct Die {
    int x_center, y_center, x_min, y_min, x_max, y_max, die_flag_1, die_flag_2, die_flag_3, die_flag_4;
};

nlohmann::json read_json_file(
    std::string file_path
    );

std::tuple<cv::Mat, std::vector<std::vector<cv::Point>>> retrive_contours(
    cv::String img_path,
    std::unordered_map<std::string, int> args
    );

std::tuple<std::vector<Die>, std::vector<Die>> contours2rects(
    const std::vector<std::vector<cv::Point>>& contours,
    std::unordered_map<std::string, int> args
    );

#endif // CONTOURS_H