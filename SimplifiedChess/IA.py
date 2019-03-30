import numpy as np
import random

# Hyperparameters
learningRate = 0.1
randomFactor = 0.1
rewardForKill = 2
enemyFactor = 0.8
initialQValue = 1

# Actions are (elementId, directionId)
actions = []
for i in range(8):
    actions.append((1, i))
for i in range(4):
    actions.append((2, i))
for i in range(8):
    actions.append((3, i))

print("Q Table initializing. Total elements:", 1000000 * len(actions))
# qTable = np.memmap("./qTable", dtype='float32', mode='w+', shape=(1000000, len(actions)))
qTable = np.full([1000000, len(actions)], initialQValue)
print("Q Table initialized")

elementMapping = [1, 2, 3, -1, -2, -3]
def getStateFromBoard(board):
    total = 0
    flatBoard = board.flatten()
    for element in range(len(elementMapping)):
        where = np.nonzero(flatBoard == elementMapping[element])[0]
        if (len(where) != 0):
            # where[0] + 1 because a value of zero represents a missing element, hence calculating total in base 10 (9 tiles + missing)
            total += (where[0] + 1) * (10 ** element)
    # Total is a number up to 999.999. All values in between arent used bc there cant be two pieces on the same tile
    # and there is always at least one king (technically cases with only to kings arent used either), but this result is good enough
    return total

def valueFunction(response, enemyState):
    if (response == None):
        return -10000
    if (abs(response) == 1):
        return +10000
    # You lose points depending on what you opponent can expect
    enemyMax = np.max(qTable[enemyState])
    return (response != 0) * rewardForKill - enemyFactor * enemyMax

def playATurn(game, player, training):
    # game: a Game instance, including the play and getBoard methods
    # player: +1 or -1, representing who is currently playing
    # training: wether to use random factor and display the result
    state = getStateFromBoard(game.getBoard() * player)
    actionId = None
    if ((random.uniform(0, 1) < randomFactor) & training == True):
        actionId = np.random.randint(len(actions))
    else:
        actionId = np.argmax(qTable[state])
    action = actions[actionId]
    response = game.play(player, action[0], action[1])

    enemyState = getStateFromBoard(game.getBoard() * player * (-1))
    value = valueFunction(response, enemyState)
    qTable[state, actionId] = (1 - learningRate) * qTable[state, actionId] + learningRate * value

    if (response == None):
        return False
    if (abs(response) == 1):
        return False
    return True
