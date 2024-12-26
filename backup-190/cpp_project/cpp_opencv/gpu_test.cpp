#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/cuda.hpp"
using namespace cv;
using namespace std;

int main() {
    int num_devices = cv::cuda::getCudaEnabledDeviceCount();

    if (num_devices <= 0) {
        std::cerr << "There is no device." << std::endl;
        return -1;
    }
    int enable_device_id = -1;
    for (int i = 0; i < num_devices; i++) {
        cv::cuda::DeviceInfo dev_info(i);
        if (dev_info.isCompatible()) {
            enable_device_id = i;
        }
    }
    if (enable_device_id < 0) {
        std::cerr << "GPU module isn't built for GPU" << std::endl;
        return -1;
    }
    cv::cuda::setDevice(enable_device_id);

    std::cout << "GPU is ready, device ID is " << num_devices << "\n";

    return 0;
}
