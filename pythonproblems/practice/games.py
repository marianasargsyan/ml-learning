import random

def guess_number():

    number = random.randint(1, 9)
    guess = 0
    count = 0

    while guess != number and guess != "exit":
        guess = input("Please guess the number: ")

        if guess == "exit":
            break

        guess = int(guess)
        count += 1

        if guess == number:
            print("Your guess is correct!")
            print("You tried: ", count, "times.")
        else:
            print("Please guess again")



def birthday_check(dct):
    print('Welcome to the birthday dictionary. We know the birthdays of:')
    for name in dct:
        print(name)

    print("Who's birthday do you want to look up?")
    name = input()
    if name in dct:
        print('{}\'s birthday is {}.'.format(name, dct[name]))
    else:
        print('Sadly, we don\'t have {}\'s birthday.'.format(name))


if __name__ == '__main__':
    guess_number()
    birthdays = {
        'Albert Einstein': '03/14/1879',
        'Benjamin Franklin': '01/17/1706',
        'Ada Lovelace': '12/10/1815',
        'Donald Trump': '06/14/1946',
        'Rowan Atkinson': '01/6/1955'}

    birthday_check(birthdays)

