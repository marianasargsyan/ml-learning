from random import randint


def sorter(lst):
    arr = [0]*(max(lst) + 1)
    for number in lst:
        arr[number] += 1
    out_string = ''
    for i in range(len(arr)):
        if arr[i] != 0:
            out_string += (str(i) + ', ') * arr[i]
    print(out_string)


if __name__ == '__main__':
    lst1 = []
    for _ in range(0, 10):
        lst1.append(randint(0, 50))
    sorter(lst1)
