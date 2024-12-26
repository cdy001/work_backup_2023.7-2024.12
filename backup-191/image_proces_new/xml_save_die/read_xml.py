import xml.etree.cElementTree as ET


# 读取xml文件，返回缺陷字典
def read_xml(xml_path):
    print(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    defect_list = []
    xy_list = []
    for objects in root.findall('object'):
        xmin1 = int(float(objects.find('bndbox').find('xmin').text))
        ymin1 = int(float(objects.find('bndbox').find('ymin').text))
        xmax1 = int(float(objects.find('bndbox').find('xmax').text))
        ymax1 = int(float(objects.find('bndbox').find('ymax').text))
        defect_name = objects.find('name').text

        defect_list.append(defect_name)

        xy_list.append((int(xmin1), int(ymin1), int(xmax1), ymax1))

    return defect_list, xy_list
