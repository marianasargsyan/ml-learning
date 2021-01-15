def check_fermat(a, b, c, n):
    if n <= 2:
        print("Try grater n")
    if n > 2 and a ^ n + b ^ n == c ^ n:
        print("Holy smokes, Fermat was wrong!")
    if n > 2 and a ^ n + b ^ n != c ^ n:
        print('No, that doesn’t work.')


check_fermat(1, 2, 3, 3)

def fermat_input():
    a = int(input("Type a: "))
    b = int(input("Type b: "))
    c = int(input("Type c: "))
    n = int(input("Type n: "))
    return check_fermat(a, b, c, n)

fermat_input()
