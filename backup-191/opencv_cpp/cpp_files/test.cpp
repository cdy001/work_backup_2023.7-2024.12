#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;


json read_json_file(std::string file_path) {
    // 打开 JSON 文件
    std::ifstream j_file(file_path);
    if (!j_file.is_open()) {
        std::cerr << "Failed to open the file!" << std::endl;
    }
    // 解析 JSON 文件
    json json_dict;
    j_file >> json_dict;

    return json_dict;
}

int main(){
    std::string file_path = "/data/chendeyang/code/torch-tensorrt/configs/aoyang_yuanxing_1.json";
    
    json json_dict = read_json_file(file_path);

    for (const auto& entry: json_dict.items()) {
        std::cout << entry.key() << ": " << entry.value() << std::endl;
    }

    return 0;
}