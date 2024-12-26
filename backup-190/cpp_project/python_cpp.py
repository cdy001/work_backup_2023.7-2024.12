import subprocess
import time


if __name__ == "__main__":
    time_start = time.time()
    # 调用 C++ 可执行文件
    result = subprocess.run(["./hello_world"], capture_output=True, text=True)
    time_end = time.time()
    print(f"Cpp time cost: {time_end - time_start: .4f}s")
    print("Hello World!")
    print(f"Python time cost: {time.time()- time_end: .4f}s")

    # # 打印 C++ 程序的输出
    # print("C++ Program Output:", result.stdout)
