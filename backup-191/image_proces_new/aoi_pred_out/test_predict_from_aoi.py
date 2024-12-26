import glob
import os.path
import shutil


def get_new_predict_out_file(aoi_out_file_path, sort_die_file_path, predict_out_file_path, new_predict_file_path):
    aoi_label_dict = {}
    if os.path.isfile(aoi_out_file_path):
        with open(aoi_out_file_path, "r") as r:
            for line in r.readlines():
                if len(line.split(",")) >= 5:
                    line_str_list = line.split(",")
                    original_x = line_str_list[0]
                    original_y = line_str_list[1]
                    original_label = line_str_list[2]
                    aoi_label_dict[f"{original_x}:{original_y}"] = original_label

        predict_image_label_dict = {}
        with open(sort_die_file_path, "r") as r:
            for die_data in r.readlines():
                x, y, predict_location = die_data.split(":")
                aoi_label = aoi_label_dict.get(f"{x}:{y}", "13")
                image_name, seq = predict_location.split("_")[:2]
                predict_image_label_dict[f"{image_name}_{seq}"] = aoi_label

        with open(predict_out_file_path, "r") as r, open(new_predict_file_path, "a") as w:
            for die_data in r.readlines():
                location = die_data.split(",")[0]
                image_name, seq = location.split("_")[:2]
                label = predict_image_label_dict.get(f"{image_name}_{seq}", "13")
                w.write(location + "," + label + "\n")


def get_differ_out_file(aoi_out_file_path, ai_out_file_path, result_file_path):
    aoi_label_dict = {}
    with open(aoi_out_file_path, "r") as r:
        for line in r.readlines():
            if len(line.split(",")) >= 5:
                line_str_list = line.split(",")
                original_x = line_str_list[0]
                original_y = line_str_list[1]
                original_label = line_str_list[2]
                aoi_label_dict[f"{original_x}:{original_y}"] = original_label

    miss_count = 0
    over_count = 0
    with open(ai_out_file_path, "r") as r, open(result_file_path, "a") as w:
        for line in r.readlines():
            if len(line.split(",")) >= 5:
                line_str_list = line.split(",")
                x = line_str_list[0]
                y = line_str_list[1]
                label = line_str_list[2]
                key = f"{x}:{y}"
                aoi_label = aoi_label_dict.get(key, "0")

                if aoi_label in ["150", "151"] and label not in ["150", "151"]:
                    label = "miss"
                    miss_count += 1
                elif aoi_label not in ["150", "151"] and label in ["150", "151"]:
                    label = "over"
                    over_count += 1
                line_str_list[2] = label
                w.write(",".join(line_str_list).strip() + "\n")
            else:
                w.write(line)
    print("miss:", miss_count, "over:", over_count)


def aoi_predict(test_dir):
    # test_dir = "/data/wz/data/data/32BB/08.26/result/"
    for wafer_dir in glob.glob(os.path.join(test_dir, "*-*")):
        print(f"start to handle:{wafer_dir}")
        wafer_id = os.path.split(wafer_dir)[-1]
        result_dir = os.path.join(test_dir, "result2", wafer_id)
        print("result_dir:", result_dir)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)

        aoi_out_file_path = os.path.join(wafer_dir, f"{wafer_id}_aoi.txt")
        sort_die_file_path = os.path.join(wafer_dir, "in_sort_die.txt")
        predict_out_file_path = os.path.join(wafer_dir, "predict_out.txt")
        new_predict_file_path = os.path.join(result_dir, "predict_out.txt")
        get_new_predict_out_file(aoi_out_file_path, sort_die_file_path, predict_out_file_path, new_predict_file_path)

        ai_out_file_path = os.path.join(wafer_dir, f"{wafer_id}.txt")
        differ_out_file_path = os.path.join(wafer_dir, f"{wafer_id}_miss_over.txt")
        differ_predict_out_file_path = os.path.join(result_dir, f"miss_over_predict_out.txt")
        # get_differ_out_file(aoi_out_file_path, ai_out_file_path, differ_out_file_path)


if __name__ == '__main__':
    aoi_predict()
