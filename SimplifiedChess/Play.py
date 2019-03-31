from Game import Game
from IA import playATurn

def playGame(training):
    game = Game()
    player = 1
    notDone = True
    count = 0

    while (notDone & (count < 200)):
        count += 1
        response = playATurn(game, player, training)
        if (response == "ok"):
                player = player * (-1)
        if (response == "end"):
                notDone = False
        if (training == False):
            print("\n")
            print(game.getBoard())

def train():
    for i in range(100):
        print("Playing game :", i)
        playGame(True)

def spectate_game():
    playGame(False)

train()
spectate_game()
