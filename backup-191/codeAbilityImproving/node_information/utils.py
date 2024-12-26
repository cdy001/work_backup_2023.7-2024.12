class DataProcessor:
    def __init__(self, node):
        self.node = node

    # 采集
    def collect_raw_data(self):
        '''
        这个函数用于开发数据采集功能
        '''
        new_data = self.node.raw_data
        return new_data

    # 转换数据
    def transform_data(self, raw_data):
        '''
        这个函数用于开发数据转换功能
        '''
        transformed_data = {}
        for key, value in raw_data.items():
            # 转换、加工等操作
            transformed_data[key] = value * 2
        return transformed_data

    # 数据生成
    def generate_final_data(self, transformed_data):
        '''
        这个函数用于开发数据生成功能
        '''
        final_data = {}
        for key, value in transformed_data.items():
            # 进一步处理、生成最终数据
            final_data[key] = value + 10
        return final_data
