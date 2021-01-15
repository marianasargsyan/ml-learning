#Create Queue

class Queue:
    def __init__(self, init_size=100):
        self.lst = [0] * init_size
        self.index = -1

    def push(self, data):
        self.index += 1
        if self.index >= len(self.lst):
            print("enlarging stack")
            temp_list = self.lst
            self.lst = [0] * 2 * len(self.lst)
            self.lst[0:len(temp_list)] = temp_list
        self.lst[self.index] = data

    def pop(self):
        if len(self.lst) == 0:
            print("List is empty")
            exit(0)
        else:
            out = self.lst[0]
            if self.index-1 >= 0:
                self.lst[0:self.index] = self.lst[1:self.index + 1]
            self.index -= 1
            return out

    def front(self):
        return self.lst[0]

    def size(self):
        return len(self.lst)



if __name__ == '__main__':
    q = Queue(50)
    for i in range(0, 100):
        q.push(i)
    print("Front", q.front())
    print("Size", q.size())
    for i in range(0, 100):
        print(q.pop())