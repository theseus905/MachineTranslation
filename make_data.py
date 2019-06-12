import random
import numpy as np

def make_repeat():
    file  = open("data/repeat.txt", "w")
    for row in range(2000):
        rand = random.randint(0,19)
        for i in range(10):
            file.write(str(rand))
            file.write(" ")
        file.write("\n")

def make_random():
    file  = open("data/random.txt", "w")
    for row in range(2000):
        for i in range(10):
            rand = random.randint(0,19)
            file.write(str(rand))
            file.write(" ")
        file.write("\n")

def make_highnums():
    file  = open("data/highnums.txt", "w")
    for row in range(2000):
        for i in range(10):
            rand = random.randint(20,50)
            file.write(str(rand))
            file.write(" ")
        file.write("\n")

def make_weighted():
    file  = open("data/weighted.txt", "w")
    for row in range(2000):
        for i in range(10):
            low_rand = random.randint(0,19)
            high_rand = random.randint(20,50)
            options = [low_rand,high_rand]
            weights = [0.7,0.3]
            file.write(str(np.random.choice(options, p=weights)))
            file.write(" ")
        file.write("\n")

make_repeat()
make_random()
make_highnums()
make_weighted()
