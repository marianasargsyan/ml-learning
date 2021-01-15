def factorial(n):
    global val
    if n < 1:
        return 1
    else:
        val = n * factorial(n - 1)
    return val


for n in range(1, 100):
    print(factorial(n))
