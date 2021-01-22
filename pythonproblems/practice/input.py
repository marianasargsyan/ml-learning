def get_age_in100():
    name = input("Please enter your name: ")
    age = int(input("Please enter your age: "))
    year = str((2021 - age) + 100)
    print(name + " will be 100 years old in  " + year)


def check_palindrome():
    name = input("Enter your word: ")
    if name[::-1] == name[0:]:
        print(name, " is  palindrome")
    else:
        print(name, " is not a palindrome")


if __name__ == '__main__':
    get_age_in100()
    check_palindrome()
