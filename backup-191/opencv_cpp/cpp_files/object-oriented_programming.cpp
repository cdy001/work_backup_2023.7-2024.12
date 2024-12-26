#include <iostream>

using namespace std;


class MyClass {
    int z;
    public:
        int myNum;
        string myString;
        MyClass(int x, string y);
    void myMethod();
    int speed(int maxSpeed);
};

void MyClass::myMethod()  {
    cout << "Hello World!" << endl;
};

int MyClass::speed(int maxSpeed) {
    return maxSpeed;
}

MyClass::MyClass(int x, string y) {
    cout << "this is a example of constructor in class." << endl;
    myNum = x;
    myString = y;
}

class Employee {
    private:
        int salary;
    
    public:
        void setSalary(int s) {
            salary = s;
        };
        int getSalary() {
            return salary;
        };
};

// base class
class Vehicle {
    public:
        string brand = "Ford";
        void honk() {
            cout << "Tuut, tuut!" << endl;
        }
};

// derived class
class Car: public Vehicle {
    public:
        string model = "Mustang";
};

class BaseClass_1 {
    public:
        void myFunction_1() {
            cout << "this is a example of function_1." << endl;
        }
};

class BaseClass_2 {
    public:
        void myFunction_2() {
            cout << "this is a example of function_2." << endl;
        }
};

class MultiDerivedClass: public BaseClass_1, public BaseClass_2 {

};

int main() {
    MultiDerivedClass myclass;
    myclass.myFunction_1();
    myclass.myFunction_2();
    return 0;
}