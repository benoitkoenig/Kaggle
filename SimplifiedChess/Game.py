import numpy as np

MOVES = [
    None,
    [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]], # 1 is the King
    [[-1, -1], [1, -1], [-1, 1], [1, 1]], # 2 is the Fou
    [[-1, -2], [1, -2], [-1, 2], [1, 2], [-2, -1], [-2, 1], [2, -1], [2, 1]], # 3 is the Horse
]

class Game:
    def __init__(self):
        self.board = np.matrix([[1, 2, 3], [0, 0, 0], [-1, -2, -3]]).A

    def play(self, player, number, direction):
        if (direction > len(MOVES[number])):
            return None
        element = player * number # player is +1 or -1 and number is always absolute

        whereIs = np.where(self.board == element)
        if (len(whereIs[0]) == 0): # If trying to move a piece that is already out
            return None
        currentPosition = np.reshape(whereIs, 2)

        newPosition = currentPosition + MOVES[number][direction]
        if (any(((c < 0) | (c > 2)) for c in newPosition)):
            return None

        deleted = self.board[newPosition[0]][newPosition[1]]
        if (player * deleted > 0): # You cant kill your own piece
            return None

        self.board[newPosition[0]][newPosition[1]] = element
        self.board[currentPosition[0]][currentPosition[1]] = 0
        return deleted

    def getBoard(self):
        return self.board
