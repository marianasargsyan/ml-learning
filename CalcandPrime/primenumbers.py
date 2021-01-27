import numpy as np


def get_prime():
    n = int(input("Please enter a number up to 100: "))

    if n > 1000:
        print("Please enter a number up to 100")
    else:
        full_list = list(range(2, n))
        int_list = list(range(2, n//10))
        tmp = []
        for number in full_list:
            for integer in int_list:
                if number != integer and number % integer == 0:
                    tmp.append(number)
        prime_list = set(full_list) - set(tmp)

        return np.sort(list(prime_list))


if __name__ == "__main__":
    print(get_prime())
