import random
import numpy as np
import matplotlib.pyplot as plt


def coin_flip():
    return random.randint(0, 1)


print(coin_flip())

# The simulation

#Create an empty list to store the probability values.
list_1 = []

def monte_carlo(n):
    results = 0
    for i in range(n):
        flip_result = coin_flip()
        results = results + flip_result

        #Calculating probability value:
        prob_value = results/(i+1)

        #Append the probability values to the list:
        list_1.append(prob_value)

        #Plot the results:
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.plot(list_1)
    plt.show()

    return results/n

if __name__ == '__main__':
    answer = monte_carlo(5000)
    print("final value :", answer)



