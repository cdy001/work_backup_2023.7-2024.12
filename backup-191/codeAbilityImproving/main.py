from node_information.node import Node1, Node2, collect_node_information

def main():
    node1, node2 = Node1(), Node2()
    node1_raw_data = node1.raw_data
    node2_raw_data = node2.raw_data
    result1 = collect_node_information(node1)
    print(f"node1_raw_data: {node1_raw_data}\nnode1_result: {result1}")
    print(f"------------------------------------------------")
    result2 = collect_node_information(node2)
    print(f"node2_raw_data: {node2_raw_data}\nnode2_result: {result2}")
    print(f"------------------------------------------------")

    # 升级迭代Node1
    class Node1_new(Node1):
        def __init__(self, node_name="node1", raw_data=...) -> None:
            super().__init__(node_name, raw_data)
        # 添加"采集节点信息"的方法
        def collect_information(self, node):
            result = collect_node_information(node)
            return result
    node1_new = Node1_new()
    print(f"node1_new_information: {node1_new.collect_information(node1)}")

if __name__ == "__main__":
    main()