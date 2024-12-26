import os
import json

root_path = r'08_03.09'

all_classes = [cla for cla in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, cla))]
all_classes.sort()
# print(all_classes)

class_indices = dict((k, v) for v, k in enumerate(all_classes))

json_str = json.dumps(dict((key, val) for key, val in class_indices.items()), indent=4)
with open('label_08.txt', 'w') as json_file:
    json_file.write(json_str)