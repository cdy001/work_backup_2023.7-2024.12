#include <iostream>

using namespace std;


int max(int num1, int num2);

int main(int argc, char **argv) {
    int a = 0;
    int b = 0;
    cout << "请输入需要比较大小的两个数: ";
    cin >> a >> b;
    cout << "更大的数是: " << max(a, b) << endl;

    return 0;
}

// function returning the max between two numbers
int max(int num1, int num2) {
   // local variable declaration
//    int result;
 
//    if (num1 > num2)
//       result = num1;
//    else
//       result = num2;
 
//    return result; 
    
    return num1 > num2 ? num1 : num2;
}

