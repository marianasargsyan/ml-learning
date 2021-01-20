def get_less_5(lst):
    output = []
    for i in lst:
        if i < 5:
            output.append(i)
    return output


def duplicate_check(lst1,lst2):
    dup_lst = []
    for i in lst1:
        for j in lst2:
            if i == j:
                dup_lst.append(i)
    return dup_lst


def return_evens(lst):
    output = []
    for i in lst:
        if i % 2 == 0:
            output.append(i)
    return output


if __name__ == '__main__':
    a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    print(get_less_5(a))

    b = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print(duplicate_check(b,c))

    e = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    print(return_evens(e))