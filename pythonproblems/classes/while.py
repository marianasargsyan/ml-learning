for i in range(0, 11):
    print(i)

i = 0
while i <= 10:
    print(i)
    i += 1

lastNumber = 6
for row in range(1, lastNumber):
    for column in range(1, row + 1):
        print(column, end=' ')
    print("")


def sum_num(n):
    n = sum(list(range(0, n + 1)))
    print(n)


sum_num(10)


def mul_1(n):
    for i in range(1, 11):
        prod = n * i
        print(prod)


mul_1(2)

list1 = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]


def by5(lst):
    for i in lst:
        if i > 150:
            break
        if i % 5 == 0:
            print(i)


by5(list1)
