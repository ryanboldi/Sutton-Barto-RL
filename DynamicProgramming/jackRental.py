from math import exp, factorial

facts = [factorial(n) for n in range(0, 30)]

def poisson(lam, n):
    return ((lam**n)/(facts[n]))*exp(-lam)

class rentalPolicy:
    def __init__(self):
        self.carPark1 = 20
        self.carPark2 = 20
        self.actions = [[0 for i in self.carPark2] for k in self.carPark2]

    def eval(self):
        delta = 0
        for i in self.actions:
            for j in i:
                pass


class rentalEnv1:
    def __init__(self):
        self.cars = 10
        self.stateValues = [[0 for i in self.cars] for k in self.cars] #how good each state is

    def valueOfState(self, carsLot1, carsLot2, action):
        #if out of cars, the business is lost
        if carsLot1 == 0 or carsLot2 == 0:
            return -1000000

        #10 bucks for every rented car
        rew = 0

        carsLot1 -= action
        carsLot2 += action

        rew += 10*rented


        #-2 for every moved car
        rew -= 2*action
