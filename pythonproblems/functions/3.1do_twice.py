def do_twice(f):
    f()
    f()

def do_four(f):
    do_twice(f)
    do_twice(f)


