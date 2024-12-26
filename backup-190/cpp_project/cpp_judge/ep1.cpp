#include <iostream>

enum Week 
{
    Mon, // 星期一
    Tue, // 星期二
    Wed, // 星期三
    Thu, // 星期四
    Fri, // 星期五
    Sat, // 星期六
    Sun, // 星期日
};

int main(int argc, char **argv){
    // Week week = Week::Mon;
    // Week week = Week::Fri;
    int week = 0;
    std::cin >> week;

    if(week == Week::Mon){
        std::cout << Week::Mon << std::endl;
    }
    else if(week == Week::Tue){
        std::cout << Week::Tue << std::endl;
    }
    else if(week == Week::Wed){
        std::cout << Week::Wed << std::endl;
    }
    else if(week == Week::Thu){
        std::cout << Week::Thu << std::endl;
    }
    else if(week == Week::Fri){
        std::cout << Week::Fri << std::endl;
    }
    else if(week == Week::Sat){
        std::cout << Week::Sat << std::endl;
    }
    else{
        std::cout << Week::Sun << std::endl;
    }
    // std::cout << Week::Sun << std::endl;.

    return 0;
}