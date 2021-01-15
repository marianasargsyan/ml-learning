# make list
# Generate 100 random numbers
# add to list
# make class node
# make class bst

# # dct = dict(a=2, b=4)
# dct = {0: lst[0]}
# print(dct[0])
# exit(0)

import random


def create_dict(lst):
    d = {}
    for l in lst:
        i = 0
        while i in d:
            if l < d[i]:
                i = 2 * i + 1
            else:
                i = 2 * i + 2
        d[i] = l
    return d


def find(dct, val):
    k = []
    i = 0
    while i in dct:
        if val > dct[i]:
            i = 2 * i + 2
        elif val < dct[i]:
            i = 2 * i + 1
        else:
            k.append(i)
            i = 2 * i + 2
    return k


def find_2(dct, value):
    k = []
    for key, v in dct.items():
        if value == v:
            k.append(key)
    return k


lst1 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10]
# for i in range(50):
#     lst.append(random.randint(1, 1000))
d = create_dict(lst1)

f = find(d, 8)
print(f)
