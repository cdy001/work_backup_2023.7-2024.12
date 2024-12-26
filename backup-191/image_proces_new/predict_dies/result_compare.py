import os
from collections import defaultdict


def read_result(result_file):
    result_dict = defaultdict()
    with open(result_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            path, label = line.split(",")
            name = os.path.split(path)[-1]
            if name not in result_dict:
                result_dict[name] = int(label)
            else:
                # print(name, result_dict[name] == int(label))
                continue
    
    return result_dict

if __name__ == "__main__":
    result_3090 = "predict_dies/result-3090.txt"
    result_4080s = "predict_dies/result-4080s.txt"
    result_30 = read_result(result_3090)
    result_40 = read_result(result_4080s)

    number = 0
    ng_ok_error = 0
    mark_error = 0
    for name, label in result_30.items():
        label_40 = result_40.get(name, None)
        if label_40 != label:
            print(name, label, label_40)
            number += 1
            if 0 in [label, label_40]:
                ng_ok_error += 1
            if label == 1 or label_40 == 1:
                mark_error += 1
    print(f"total_diff: {number}, ng_ok_error: {ng_ok_error}, mark_error: {mark_error}")