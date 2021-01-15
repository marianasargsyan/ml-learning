# create list

import copy


class Node:
    def __init__(self):
        self.data = None
        self.next = None


class List:
    def __init__(self):
        self.node = Node()
        self.initial = None
        self.size = 0

    def insert(self, d):
        self.size += 1
        if self.node.data is None:
            self.node.data = d
            self.initial = self.node
        else:
            temp_node = Node()
            temp_node.data = d
            self.node.next = temp_node

    def index(self, index):
        if index <= self.size:
            temp = copy.deepcopy(self.initial)
            print(temp.next.data)
            for _ in range(index):
                temp = temp.next
            return temp.data


l1 = List()
l1.insert(5)
l1.insert(1)
print(l1.index(2))

