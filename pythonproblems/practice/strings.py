def reverse_word(x):
    y = x.split()
    return " ".join(y[::-1])



if __name__ == '__main__':
    print(reverse_word("Hello my dear world"))