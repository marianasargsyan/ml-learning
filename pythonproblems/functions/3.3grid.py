def do_twice(f):
    f()
    f()

def do_four(f):
    do_twice(f)
    do_twice(f)

def ceil():
    print('+ - - - -', end=' ')

def walls():
    print('|        ', end=' ')

def plus():
    do_twice(ceil)
    print('+')

def straight():
    do_twice(walls)
    print('|')

def row():
    plus()
    do_four(straight)

def grid():
    do_twice(row)
    plus()

grid()
