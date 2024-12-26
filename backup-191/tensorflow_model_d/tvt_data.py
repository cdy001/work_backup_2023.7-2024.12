import csv
import glob
import json
import os
import random
import shutil


class DatasetSplit:
    def __init__(self, data_path, dest_path, label_path):
        self.data_path = data_path
        self.dest_path = dest_path
        self.label_path = label_path

        self.scale = 0.1

        self.test_validation = True
        self.is_copy = True
        self.is_write = False

    # 列表划分
    @staticmethod
    def list_tvt(label_images, scale, test_validation):
        random.shuffle(label_images)
        images_numbers = len(label_images)
        n = int(images_numbers * scale)

        if test_validation:
            test = label_images[:n]
            validation = test
            train = label_images[n:] 
        else:
            test = label_images[:n]
            validation = label_images[n:2 * n]
            train = label_images[2 * n:]
            

        return test, validation, train

    # 读取标签文件
    @staticmethod
    def read_label(label_path):

        with open(label_path, 'r') as f:
            label_dict = json.load(f)

        return label_dict

    # 移动文件图片
    @staticmethod
    def copy_img(img_list, label_num, save_path, tvt):
        label_folder_path = os.path.join(save_path, tvt, str(label_num))
        if not os.path.exists(label_folder_path):
            os.makedirs(label_folder_path)
        for img in img_list:
            shutil.copy(img, label_folder_path)

    # csv文件写入
    @staticmethod
    def write_csv(img_list, save_path, label_dict, tvt):
        header = ['img_path', 'label']
        csv_path = os.path.join(save_path, tvt + '.csv')
        with open(csv_path, mode='w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for img_path in img_list:
                name = img_path.split(os.sep)[-2]
                label = label_dict[name]

                writer.writerow([img_path, label])

        print(f"writen into csv file: {csv_path}")

        return csv_path

    # 数据划分
    def tvt(self):
        data_path = self.data_path
        dest_path = self.dest_path
        label_path = self.label_path
        scale = self.scale

        print(f"total image num: {len(glob.glob(os.path.join(data_path, '*', '*', '*bmp')))}")

        # 读取标签对应文件
        label_dict = DatasetSplit.read_label(label_path)

        test_list = []
        validation_list = []
        train_list = []
        label_number = 0
        for label_dict_key in label_dict:
            label_dict_value = label_dict[label_dict_key]

            label_image_path = os.path.join(data_path, '*', label_dict_key, '*bmp')
            label_images = glob.glob(label_image_path)
            print(f"{label_dict_key}: {len(label_images)}")
            if len(label_images) > 0:
                label_number += 1

            test, validation, train = DatasetSplit.list_tvt(label_images, scale, self.test_validation)

            # 写入列表
            test_list.extend(test)
            validation_list.extend(validation)
            train_list.extend(train)

            # 是否复制图片
            if self.is_copy:
                DatasetSplit.copy_img(test, label_dict_value, dest_path, 'test')
                DatasetSplit.copy_img(validation, label_dict_value, dest_path, 'validation')
                DatasetSplit.copy_img(train, label_dict_value, dest_path, 'train')

        # 是否写入csv文件
        random.shuffle(test_list)
        random.shuffle(validation_list)
        random.shuffle(train_list)
        print(f"number of test img: {len(test_list)}")
        print(f"number of validation img: {len(validation_list)}")
        print(f"number of train img: {len(train_list)}")

        train_csv_path, validation_csv_path, test_csv_path = None, None, None
        if self.is_write:
            test_csv_path = DatasetSplit.write_csv(test_list, dest_path, label_dict, 'test')
            validation_csv_path = DatasetSplit.write_csv(validation_list, dest_path, label_dict, 'validation')
            train_csv_path = DatasetSplit.write_csv(train_list, dest_path, label_dict, 'train')

        print(f"label_number: {label_number}")
        return train_csv_path, validation_csv_path, test_csv_path


if __name__ == '__main__':
    # 路径
    data_path_s = '/var/cdy_data/aoyang/data/BGB1Q42E-D'
    dest_path_s = '/var/cdy_data/aoyang/data/BGB1Q42E-D_tvt'
    label_path_s = 'model_and_label/BOB1P42M.txt'
    # label_path_s = 'model_and_label/3535.txt'

    dataset = DatasetSplit(data_path_s, dest_path_s, label_path_s)
    _, _, _ = dataset.tvt()
