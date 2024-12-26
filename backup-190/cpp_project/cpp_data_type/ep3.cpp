#include <iostream>

struct Student{
    int math = 0;
    int english = 0;
};

int main(int argc, char **argv){
    struct Student stu[50];
    
    //为其中一个学生的成绩赋值
    stu[20].math = 98;
    stu[20].english = 82;

    
    std::cout << stu[20].english << std::endl;
    
    return 0;
}