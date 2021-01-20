def check_odd_even(num):
    if num % 2 == 0:
        return "even"
    return "odd"


def divisors(n):
    lst = []
    for i in range(1,n):
        if n % i == 0:
            lst.append(i)
    return lst


def max_of_three(a, b, c):
    max = 0
    if a > b:
        if a > c:
            max = a
        else:
            max = c
    else:
        if b > c:
            max = b
        else:
            max = c
    return max

if __name__ == '__main__':

    print(check_odd_even(5))
    print(check_odd_even(56))
    print(divisors(10))

    print(max_of_three(3,6,8))