import random

_line = open('sowpods.txt', 'r')
lines = [(line.strip().split()) for line in _line]
_line.close()


def pick_word(word_list):
    word = random.choice(word_list)
    return word


if __name__ == '__main__':
    word = pick_word(lines)
    print(word)

    tmp = ['_'] * len(word[0])
    lst_guessed = []
    guessed = 0
    letter = 0
    count = len(word[0])

    print('Welcome to Hangman!')

    while count != 0 and letter != "exit":
        print(''.join(tmp))
        letter = input("Guess a letter: ")

        if letter in lst_guessed:
            letter = ''
            print('Already guessed')

        else:
            if letter in word[0]:
                index = word[0].index(letter)
                print('Correct guess!')
                tmp[index] = letter
                word[index] = '_'
                guessed += 1
            count -= 1

    lst_guessed.append(letter)

    if letter == "exit":
        print("Thanks!")
    else:
        print(" ".join(tmp))
        print("Good job! You figured that the word is " + word + " after guessing " + guessed + " letters!")
