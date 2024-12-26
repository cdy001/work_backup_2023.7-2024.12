#include <stdio.h>
#include <iostream>

int main(int argc,char **argv)
{
    // printf("char: %d\n", sizeof(char));
    // printf("unsigned char: %d\n", sizeof(unsigned char));

    // printf("short: %d\n", sizeof(short));
    // printf("unsigned short: %d\n", sizeof(unsigned short));

    // printf("int: %d\n", sizeof(int));
    // printf("unsigned int: %d\n", sizeof(unsigned int));

    // printf("long: %d\n", sizeof(long));
    // printf("unsigned long: %d\n", sizeof(unsigned long));

    // printf("long long: %d\n", sizeof(long long));
    // printf("unsigned long long: %d\n", sizeof(unsigned long long));

    std::cout << "char: " << sizeof(char) << std::endl;
    std::cout << "unsigned char: " << sizeof(unsigned char) << std::endl;

    std::cout << "short: " << sizeof(short) << std::endl;
    std::cout << "unsigned short: " << sizeof(unsigned short) << std::endl;

    std::cout << "int: " << sizeof(int) << std::endl;
    std::cout << "unsigned int: " << sizeof(unsigned int) << std::endl;

    std::cout << "long: " << sizeof(long) << std::endl;
    std::cout << "unsigned long: " << sizeof(unsigned long) << std::endl;

    std::cout << "long long: " << sizeof(long long) << std::endl;
    std::cout << "unsigned long long: " << sizeof(unsigned long long) << std::endl;

    return 0;
}