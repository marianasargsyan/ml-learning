class Account:

    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def deposit(self, v):
        if v < self.balance:
            print('dep acc')
        else:
            print('dep rej')

    def withdraw(self, v):
        pass

    def __str__(self):
        return "Account owner: " + self.owner + "\nAccount balance: $" + str(self.balance)

acc1 = Account('Jose', 100)

print(acc1)


try:
    for i in ['a','b','c']:
        print(i**2)
except:
    print("Error Message")

