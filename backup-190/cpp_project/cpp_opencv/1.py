# import os

# os.system("cpp_opencv/binary_test")

import subprocess

# 假设你的C++可执行文件是 "my_program"
result = subprocess.run(["cpp_opencv/binary_test"], capture_output=True, text=True)

# 打印程序的输出
print(result.stdout)