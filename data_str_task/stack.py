# Create stack


class Stack:
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
            last_value = self.lst[self.index]
            self.index -= 1
            return last_value

    def top(self):
        return self.lst[self.index]

    def size(self):
        return len(self.lst)


if __name__ == '__main__':
    st = Stack(50)
    for i in range(0, 100):
        st.push(i)
    print("Top", st.top())
    print("Size", st.size())
    for i in range(0, 100):
        print(st.pop())
