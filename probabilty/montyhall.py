import numpy as np
count_switch = 0 #counter for win when switch
count_stick = 0  #counter for win when stick
for i in range(10000):
    car = np.random.choice([1,2,3])    #assign door 1,2, or 3 to car randomly
    player = np.random.choice([1,2,3]) #assign door 1,2, or 3 to player selection randomly
    if car == player:
        # If Initial guess is correct and we stick, increase win numbers for 'stick'
        count_stick += 1
    else:
        # If Initial guess is incorrect and we switch, increase win number for 'switch'
        count_switch += 1

P_switch = count_switch/(count_switch+count_stick)
P_stick = count_stick/(count_switch+count_stick)

print('Win number when SWITCH:', count_switch)
print('Win probability when SWITCH:', P_switch)
print('Win number when STICK:', count_stick)
print('Win probability when STICK:', P_stick)