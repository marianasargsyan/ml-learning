import math

def sqrt_int(number):
    if number == 1:
        return 1
    num = None
    for n in range(2, round(number / 2) + 1):
        curr_val = n * n
        if curr_val == number:
            num = n
            return n
    if num is None:
        return sqrt_int(number - 1)


def sqrt_float(number):
    return pow(2, (math.log2(number))/2)


if __name__ == "__main__":
    print(sqrt_int(170))
    print(sqrt_float(5))

