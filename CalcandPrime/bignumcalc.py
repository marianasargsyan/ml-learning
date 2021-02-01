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
            result.append((lst1[i] + lst2[i]) - 10)
            result[i-2] += 1
        else:
            result.append(lst1[i] + lst2[i])
    tmp_lst = list(range(0, diff))

    rev_result = result[::-1]

    for i in tmp_lst:
        rev_result.insert(i, int(str1[i]))

    str_result = ''.join(map(str, rev_result))

    return str_result


def Sum(str1, str2):
    diff = abs(len(str1)-len(str2))
    if len(str1) < len(str2):
        str1 = '0'*diff + str1
    else:
        str2 = '0'*diff + str2

    acc = 0
    result = ''

    for i in reversed(range(len(str1))):
        tmp = int(str1[i]) + int(str2[i]) + acc
        acc = tmp // 10
        result = str(tmp % 10) + result
    return result





if __name__ == '__main__':
    tst2 = "2815345645623"
    tst1 = "16457657569"

    print(Sum(tst1, tst2), int(tst1)+int(tst2))
