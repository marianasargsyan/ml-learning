
fib_dct = {}

def fibonacci_memo(n):
    if n in fib_dct:
        return fib_dct[n]
    if n == 1:
        val = 1
    if n == 2:
        val = 1
    if n > 2:
        val = fibonacci_memo(n-1) + fibonacci_memo(n-2)
    fib_dct[n] = val

    return val

#
for n in range(1,100):
    print(fibonacci_memo(n))


def fibonacci_nomemo(n):
    if n == 1:
        return 1
    if n == 2:
        return 1
    if n > 2:
        return fibonacci_nomemo(n-1) + fibonacci_nomemo(n-2)

#
# for n in range(1,100):
#     print(fibonacci_nomemo(n))