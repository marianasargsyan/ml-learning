def findSum(str1, str2):
    if len(str1) < len(str2):
        tmp = str1
        str1 = str2
        str2 = tmp
    diff = len(str1) - len(str2)


    result = []
    lst1 = []
    lst2 = []

    rev_str1 = str1[::-1]
    rev_str2 = str2[::-1]


    for i in range(len(str1)):

        if i < len(str2):
            # result.append()
            lst1.append(int(rev_str1[i]))
            lst2.append(int(rev_str2[i]))
    for i in range(len(lst1)):
        if lst1[i] + lst2[i] >= 10:
            print(lst1[i] + lst2[i] - 10)
            result.append((lst1[i] + lst2[i]) - 10)
        else:
            result.append(lst1[i] + lst2[i])
            result[i] += 1
    tmp_lst = list(range(0, diff))

    rev_result = result[::-1]

    for i in tmp_lst:
        rev_result.insert(i, int(str1[i]))

    str_result = ''.join(map(str, rev_result))

    return str_result


if __name__ == '__main__':
    tst2 = "2226665"
    tst1 = "11199"

    print(findSum(tst1, tst2))
