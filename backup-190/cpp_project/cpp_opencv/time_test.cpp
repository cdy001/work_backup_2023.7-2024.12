#include <iostream>
#include <chrono>

using namespace std;

auto time_now()
{
    // auto now = std::chrono::system_clock::now();
    // auto duration = now.time_since_epoch();
    // auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    auto dura = chrono::system_clock::now().time_since_epoch();

    auto result = chrono::duration_cast<chrono::seconds>(dura).count();

    // cout << chrono::duration_cast<chrono::milliseconds>(dura).count() << endl;
    // cout << result << endl;

    // std::cout << "Current timestamp in milliseconds: " << millis << std::endl;

    return result;
}

int main()
{
    /*
    int time_now = time_now();
    cout << time_now << endl;
    */
    cout << time_now() <<endl;
    
    return 0;
}