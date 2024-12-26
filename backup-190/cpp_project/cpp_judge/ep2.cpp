#include <iostream>

int main(int argc, char **argv){
    int score;
    std::cout << "请输入分数: ";
    std::cin >> score;

    std::cout << "你的分数对应的评价等级是: ";
    
    if (score >= 90){
        std::cout << "优" << std::endl;
    }
    else if (75 <= score && score < 90){
        std::cout << "良" << std::endl;
    }
    else if (60 <= score && score < 75){
        std::cout << "中" << std::endl;
    }
    else{
        std::cout << "差" << std::endl;
    }
}