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





if __name__ == '__main__':
    guess_number()
