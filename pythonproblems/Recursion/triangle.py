def is_triangle(a,b,c):
    if a+b > c and a+c > b and b+c > a:
        print('Yes')
    if a+b <= c or a+c <= b or b+c <= a:
        print('No')


is_triangle(1,3,6)

is_triangle(3,4,5)

def check_triangle():
    a = int(input("Type a: "))
    b = int(input("Type b: "))
    c = int(input("Type c: "))
    return is_triangle(a, b, c)

check_triangle()