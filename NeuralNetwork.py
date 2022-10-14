import random
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        self.weightsIH = Matrix(self.hiddenNodes, self.inputNodes)
        self.weightsHO = Matrix(self.outputNodes, self.hiddenNodes)
        self.weightsIH.randomize()
        self.weightsHO.randomize()

        self.biasH = Matrix(self.hiddenNodes, 1)
        self.biasO = Matrix(self.outputNodes, 1)
        self.biasH.randomize()
        self.biasO.randomize()

        self.learningRate = 0.9

    def feedforward(self, inputList):
        inputs = Matrix.fromList(inputList)

        #calculates hidden layer
        hidden = Matrix.matrixMultiply(self.weightsIH, inputs)
        hidden.add(self.biasH)
        hidden.map(sigmoid)

        #calcultes outputs
        outputs = Matrix.matrixMultiply(self.weightsHO, hidden)
        outputs.add(self.biasO)
        outputs.map(sigmoid)

        return outputs.toList()

    def train(self, inputList, targetList):
        inputs = Matrix.fromList(inputList)

        #calculates hidden layer
        hidden = Matrix.matrixMultiply(self.weightsIH, inputs)
        hidden.add(self.biasH)
        hidden.map(sigmoid)

        #calcultes outputs
        outputs = Matrix.matrixMultiply(self.weightsHO, hidden)
        #print(outputs.data)
        outputs.add(self.biasO)
        outputs.map(sigmoid)

        targets = Matrix.fromList(targetList)

        #calculates output errors
        outputErrors = Matrix.subtract(targets, outputs)

        #calculates gradient
        gradients = Matrix.mapStatic(outputs, dsigmoid)
        gradients.multiply(outputErrors)
        gradients.multiply(self.learningRate)        

        #calculates deltas
        hiddenT = Matrix.transpose(hidden)
        weightsHODeltas = Matrix.matrixMultiply(gradients, hiddenT)

        #adjusts values by deltas
        self.weightsHO.add(weightsHODeltas)
        self.biasO.add(gradients)

        #calculates hidden layer errors
        weightsHOTranspose = Matrix.transpose(self.weightsHO)
        hiddenErrors = Matrix.matrixMultiply(weightsHOTranspose, outputErrors)

        #calculates hidden layer gradients
        gradientsH = Matrix.mapStatic(hidden, dsigmoid)
        gradientsH.multiply(hiddenErrors)
        gradientsH.multiply(self.learningRate)

        #calculates hidden layer deltas
        inputsT = Matrix.transpose(inputs)
        weightsIHDeltas = Matrix.matrixMultiply(gradientsH, inputsT)

        #adjusts values by deltas
        self.weightsIH.add(weightsIHDeltas)
        self.biasH.add(gradientsH)

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)

    def add(self, n):
        if isinstance(n, Matrix):
            
            aData = np.asarray(self.data)
            bData = np.asarray(n.data)

            self.data = np.add(aData, bData).tolist()
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    def multiply(self, n):
        if isinstance(n, Matrix):
            #hadamard product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            #scalar product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    def matrixMultiply(a, b):
        if(a.cols != b.rows):
            print("Cols of A must match rows of b")
            return None
        else:
            result = Matrix(a.rows, b.cols)
            
            aData = np.asarray(a.data)
            bData = np.asarray(b.data)

            result.data = np.dot(aData, bData).tolist()
            return result
            
    def subtract(a, b):
        result = Matrix(a.rows, a.cols)
        aData = np.asarray(a.data)
        bData = np.asarray(b.data)
        result.data = np.subtract(aData, bData).tolist()
        return result

    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[j][i] += m.data[i][j]
        return result

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-1, 1)

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val)

    def mapStatic(m, func):
        result = Matrix(m.rows, m.cols)
        for i in range(m.rows):
            for j in range(m.cols):
                val = m.data[i][j]
                result.data[i][j] = func(val)
        return result
    
    def fromList(list):
        m = Matrix(len(list), 1)
        for i in range(m.rows):
            m.data[i][0] = list[i]
        return m

    def toList(self):
        result = []  
        for i in range(self.rows):
            for j in range(self.cols):
                result.append(self.data[i][j])
        return result

"""
Types to int:
NaN: 0
Bug: 1
Dark: 2
Dragon: 3
Electric: 4
Fairy: 5
Fighting: 6
Fire: 7
Flying: 8
Ghost: 9
Grass: 10
Ground: 11
Ice: 12
Normal: 13
Poison: 14
Psychic: 15
Rock: 16
Steel: 17
Water: 18
"""

#read csv
pokemon = pd.read_csv("\pokemon.csv")
battles = pd.read_csv("combats.csv")
testData = pd.read_csv("tests.csv")

#change data to numerical values
pokemon["Legendary"] = pokemon["Legendary"].astype(int)
pokemon.fillna(0, inplace=True)
pokemon.replace("Bug", 1, inplace=True)
pokemon.replace("Dark", 2, inplace=True)
pokemon.replace("Dragon", 3, inplace=True)
pokemon.replace("Electric", 4, inplace=True)
pokemon.replace("Fairy", 5, inplace=True)
pokemon.replace("Fighting", 6, inplace=True)
pokemon.replace("Fire", 7, inplace=True)
pokemon.replace("Flying", 8, inplace=True)
pokemon.replace("Ghost", 9, inplace=True)
pokemon.replace("Grass", 10, inplace=True)
pokemon.replace("Ground", 11, inplace=True)
pokemon.replace("Ice", 12, inplace=True)
pokemon.replace("Normal", 13, inplace=True)
pokemon.replace("Poison", 14, inplace=True)
pokemon.replace("Psychic", 15, inplace=True)
pokemon.replace("Rock", 16, inplace=True)
pokemon.replace("Steel", 17, inplace=True)
pokemon.replace("Water", 18, inplace=True)

nn = NeuralNetwork(20, 40, 1)


for i in range(5000):
    x = random.randint(0, battles.shape[0]-1)
    pokemon1 = battles.iloc[x, 0] - 1
    pokemon1Type1 = pokemon.iloc[pokemon1, 2]
    pokemon1Type2 = pokemon.iloc[pokemon1, 3]
    pokemon1HP = pokemon.iloc[pokemon1, 4]
    pokemon1Attack = pokemon.iloc[pokemon1, 5]
    pokemon1Defense = pokemon.iloc[pokemon1, 6]
    pokemon1SpAttack = pokemon.iloc[pokemon1, 7]
    pokemon1SpDefense = pokemon.iloc[pokemon1, 8]
    pokemon1Speed = pokemon.iloc[pokemon1, 9]
    pokemon1Generation = pokemon.iloc[pokemon1, 10]
    pokemon1Legendary = pokemon.iloc[pokemon1, 11]

    pokemon2 = battles.iloc[x, 1] - 1
    pokemon2Type1 = pokemon.iloc[pokemon2, 2]
    pokemon2Type2 = pokemon.iloc[pokemon2, 3]
    pokemon2HP = pokemon.iloc[pokemon2, 4]
    pokemon2Attack = pokemon.iloc[pokemon2, 5]
    pokemon2Defense = pokemon.iloc[pokemon2, 6]
    pokemon2SpAttack = pokemon.iloc[pokemon2, 7]
    pokemon2SpDefense = pokemon.iloc[pokemon2, 8]
    pokemon2Speed = pokemon.iloc[pokemon2, 9]
    pokemon2Generation = pokemon.iloc[pokemon2, 10]
    pokemon2Legendary = pokemon.iloc[pokemon2, 11]

    inputs = [pokemon1Type1, pokemon1Type2, pokemon1HP, pokemon1Attack, pokemon1Defense, pokemon1SpAttack, pokemon1SpDefense, pokemon1Speed, pokemon1Generation, pokemon1Legendary, pokemon2Type1, pokemon2Type2, pokemon2HP, pokemon2Attack, pokemon2Defense, pokemon2SpAttack, pokemon2SpDefense, pokemon2Speed, pokemon2Generation, pokemon2Legendary]
    t = 0
    if battles.iloc[x, 2] == pokemon1+1:
        t = 0
    else:
        t = 1
    targets = [t]

    nn.train(inputs, targets)


inputs = []
targets = []
t = 0
x = random.randint(0, battles.shape[0]-1)

pokemon1 = battles.iloc[x, 0] - 1
pokemon1Type1 = pokemon.iloc[pokemon1, 2]
pokemon1Type2 = pokemon.iloc[pokemon1, 3]
pokemon1HP = pokemon.iloc[pokemon1, 4]
pokemon1Attack = pokemon.iloc[pokemon1, 5]
pokemon1Defense = pokemon.iloc[pokemon1, 6]
pokemon1SpAttack = pokemon.iloc[pokemon1, 7]
pokemon1SpDefense = pokemon.iloc[pokemon1, 8]
pokemon1Speed = pokemon.iloc[pokemon1, 9]
pokemon1Generation = pokemon.iloc[pokemon1, 10]
pokemon1Legendary = pokemon.iloc[pokemon1, 11]

pokemon2 = battles.iloc[x, 1] - 1
pokemon2Type1 = pokemon.iloc[pokemon2, 2]
pokemon2Type2 = pokemon.iloc[pokemon2, 3]
pokemon2HP = pokemon.iloc[pokemon2, 4]
pokemon2Attack = pokemon.iloc[pokemon2, 5]
pokemon2Defense = pokemon.iloc[pokemon2, 6]
pokemon2SpAttack = pokemon.iloc[pokemon2, 7]
pokemon2SpDefense = pokemon.iloc[pokemon2, 8]
pokemon2Speed = pokemon.iloc[pokemon2, 9]
pokemon2Generation = pokemon.iloc[pokemon2, 10]
pokemon2Legendary = pokemon.iloc[pokemon2, 11]

inputs = [pokemon1Type1, pokemon1Type2, pokemon1HP, pokemon1Attack, pokemon1Defense, pokemon1SpAttack, pokemon1SpDefense, pokemon1Speed, pokemon1Generation, pokemon1Legendary, pokemon2Type1, pokemon2Type2, pokemon2HP, pokemon2Attack, pokemon2Defense, pokemon2SpAttack, pokemon2SpDefense, pokemon2Speed, pokemon2Generation, pokemon2Legendary]
if battles.iloc[x, 2] == pokemon1+1:
    t = 0
else:
    t = 1

guess = nn.feedforward(inputs)

print(str(pokemon.iloc[battles.iloc[x, 0]-1,1]) + " vs. " + str(pokemon.iloc[battles.iloc[x, 1]-1,1]))
print("guess: " + str(guess[0]))
print("actual: " + str(t))
if(guess[0] <= 0.5):
    print("guess: " + str(pokemon.iloc[battles.iloc[x, 0]-1,1]))
else:
    print("guess: " + str(pokemon.iloc[battles.iloc[x, 1]-1,1]))
if(t == 0):
    print("actual: " + str(pokemon.iloc[battles.iloc[x, 0]-1,1]))
else:
    print("actual: " + str(pokemon.iloc[battles.iloc[x, 1]-1,1]))
