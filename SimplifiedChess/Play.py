from Game import Game
from IA import playATurn

def playGame(training):
    game = Game()
    player = 1
    notDone = True
    count = 0

    while (notDone & (count < 200)):
        count += 1
        notDone = playATurn(game, player, training)
        player = player * (-1)
        if (training == False):
            print("\n")
            print(game.getBoard())

def train():
    for i in range(10000):
        print("Playing game :", i)
        playGame(True)

def spectate_game():
    playGame(False)

train()
spectate_game()
