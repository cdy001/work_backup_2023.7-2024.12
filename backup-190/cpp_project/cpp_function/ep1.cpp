#include <iostream>

long long int fact(long long int n){
    if (n == 1){
        return 1;
    }
    else{
        return n * fact(n - 1);
    }
};

int main(int argc, char **argv){
    long long int n = 5;
    std::cout << "请输入需要计算的阶乘:";
    std::cin >> n;
    std::cout << n << "的阶乘是:" << fact(n) << std::endl;

    return 0;
}