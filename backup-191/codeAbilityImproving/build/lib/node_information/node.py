from .utils import DataProcessor

def collect_node_information(node):
    '''
    这个函数用于采集模拟节点指标信息
    '''
    data_processor = DataProcessor(node=node)
    raw_data = data_processor.collect_raw_data()
    transformed_data = data_processor.transform_data(raw_data)
    final_data = data_processor.generate_final_data(transformed_data)
    return final_data

# 信息采集基本节点
class Node():
    def __init__(self, node_name, raw_data) -> None:
        self.node_name = node_name
        self.raw_data = raw_data

# 节点1
class Node1(Node):
    def __init__(self,
                 node_name="node1",
                 raw_data={
                     'metric1': 100,
                     'metric2': 200,
                     # other metrics
                     }
                 ) -> None:
        super().__init__(node_name, raw_data)

# 节点2
class Node2(Node):
    def __init__(self,
                 node_name="node2",
                 raw_data={
                     'metric1': 150,
                     'metric2': 250
                     # other metrics
                     }
                 ) -> None:
        super().__init__(node_name, raw_data)
