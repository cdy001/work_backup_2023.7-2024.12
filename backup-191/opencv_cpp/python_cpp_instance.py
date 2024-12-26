import subprocess
import time


time_start = time.time()
result = subprocess.run(["./latest_parogram", "IMAGE1_0043.bmp", "./result_img.bmp", "configs/aoyang_yuanxing_1.json"], capture_output=True, text=True)
time_end = time.time()
print(f"time cost: {time_end - time_start: .4f}s")

print(result.stdout)
print(result.stderr)