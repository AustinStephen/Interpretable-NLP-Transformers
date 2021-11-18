## Hyperparameter tuning
## Austin Stephen

import numpy as np 

## 25 learning rates to search 
learning_rate = np.linspace(0.000001,.01,25)

## 25 epoc sizes to search prioritizing smaller epocs as they are cheaper to evaluate 
two = [2]* 2
three = [3]* 3
four = [4] * 3
five = [5] * 2
six = [6] * 2
seven = [7]
large = list(range(8,25,2))
very_large = [27,33,40]
epocs = two + three + four + five + six + seven + large + very_large

## zipping them into configurations 
configurations = zip(epocs,learning_rate)


## setting col values in file
with open("tunning_results.csv", "a") as file:
    file.write("epocs,learning_rate,accuracy\n")
    file.close()

accuracy = 100.00
for config in configurations:
    ## INSERT Collin script here.
    ## expects accuracy as output
    ## accuracy = Collin_script.py config[0] config[1]
    with open("tunning_results.csv", "a") as file: 
        file.write(str(config[0]) + "," + str(config[1]) + "," + str(accuracy)+'\n')
        file.close()
