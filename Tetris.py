import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLANK = 0

TEMPLATE_WIDTH = 5
TEMPLATE_HEIGHT = 5

S_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '..OO.',
                     '.OO..',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..OO.',
                     '...O.',
                     '.....']]

Z_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '.OO..',
                     '..OO.',
                     '.....'],
                    ['.....',
                     '..O..',
                     '.OO..',
                     '.O...',
                     '.....']]

I_SHAPE_TEMPLATE = [['..O..',
                     '..O..',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     'OOOO.',
                     '.....',
                     '.....']]

O_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '.OO..',
                     '.OO..',
                     '.....']]

J_SHAPE_TEMPLATE = [['.....',
                     '.O...',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..OO.',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '...O.',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..O..',
                     '.OO..',
                     '.....']]

L_SHAPE_TEMPLATE = [['.....',
                     '...O.',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..O..',
                     '..OO.',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '.O...',
                     '.....'],
                    ['.....',
                     '.OO..',
                     '..O..',
                     '..O..',
                     '.....']]

T_SHAPE_TEMPLATE = [['.....',
                     '..O..',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..OO.',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '..O..',
                     '.....'],
                    ['.....',
                     '..O..',
                     '.OO..',
                     '..O..',
                     '.....']]

PIECES = {'S': S_SHAPE_TEMPLATE,
          'Z': Z_SHAPE_TEMPLATE,
          'J': J_SHAPE_TEMPLATE,
          'L': L_SHAPE_TEMPLATE,
          'I': I_SHAPE_TEMPLATE,
          'O': O_SHAPE_TEMPLATE,
          'T': T_SHAPE_TEMPLATE}

PIECES_IND = {'S': 0,
              'Z': 1,
              'J': 2,
              'L': 3,
              'I': 4,
              'O': 5,
              'T': 6}

PIECES_MARGINS = {'S': [[1,1],[0,1]],
                  'Z': [[1,1],[1,0]],
                  'J': [[1,1],[0,1],[1,1],[1,0]],
                  'L': [[1,1],[0,1],[1,1],[1,0]],
                  'I': [[0,0],[2,1]],
                  'O': [[1,0]],
                  'T': [[1,1],[0,1],[1,1],[1,0]]}

class Tetris:
    def __init__(self):
        self.board = self.getBlankBoard()
        self.current_piece = self.getNewPiece()

    def reset(self):
        """
        Restarts the game with a blank board and new piece.
        Returns: torch tensor
            A tensor representing the state.

        """
        self.board = self.getBlankBoard()
        self.current_piece = self.getNewPiece()
        return self.convertToFeatures()

    def isOnBoard(self, x, y):
        """
        Checks if the position (x,y) is on the board.
        Args:
            x: int
                The x position
            y: int
                The y position

        Returns: Boolean
            If (x,y) is on the board.

        """
        return 0<=x<BOARD_WIDTH and 0<=y<BOARD_HEIGHT

    def getBlankBoard(self):
        return np.zeros((BOARD_WIDTH, BOARD_HEIGHT))

    def isValidPosition(self, x, y, rotation):
        """
        Checks if a piece has a valid position on the board.
        Args:
            shape: str
                The shape of the tetris piece.
            x: int
                The x position of the piece.
            y: int
                The y position of the piece.
            rotation: int
                The rotation of the piece.

        Returns: Boolean
            If the piece has a valid position on the board.

        """
        shape = self.current_piece
        for dx in range(TEMPLATE_WIDTH):
            for dy in range(TEMPLATE_HEIGHT):
                template = PIECES[shape][rotation % len(PIECES[shape])]
                if template[dy][dx] == 'O':
                    board_x_pos, board_y_pos = x + dx - 2, y + dy - 2
                    if not self.isOnBoard(board_x_pos, board_y_pos) or self.board[board_x_pos][board_y_pos]:
                        return False
        return True

    def getNewPiece(self):
        return random.choice(list(PIECES.keys()))

    def getNextState(self, action):
        """
        Returns the next state given the current action.
        Args:
            action: int
                An integer representing the action chosen.
                In total, there are BOARD_WIDTH x 4 actions, representing
                choices in the x coordinate and rotation of the piece.
                For a chosen x and rotation r, the action is 4 * x + r.

        Returns: tuple
            A tuple (reward, next_state, done) representing the reward, next state,
            and if the game has finished.

        """
        rotation = action % 4
        left_margin, right_margin = PIECES_MARGINS[self.current_piece][rotation % len(PIECES_MARGINS[self.current_piece])]
        x = max(left_margin, min(action // 4, BOARD_WIDTH - right_margin - 1))

        for y in range(BOARD_HEIGHT):
            if self.isValidPosition(x, y, rotation):
                self.placeOnBoard(x, y, rotation)

                lines_cleared = self.clearLines()
                delta_r, delta_c = self.countHoles()
                reward = (lines_cleared**2/16
                          - delta_r/(BOARD_HEIGHT*BOARD_WIDTH) - delta_c/(BOARD_HEIGHT*BOARD_WIDTH))/2

                self.current_piece = self.getNewPiece()
                next_state = self.convertToFeatures()
                return reward, next_state, False

        return -1, None, True

    def placeOnBoard(self, x, y, rotation):
        """
        Places the current piece on the board. Assumes that the piece
        is in a valid position.
        Args:
            x: int
                The x position of the piece.
            y: int
                The y position of the piece.
            rotation: int
                The rotation of the piece.

        Returns: None
        """
        template = PIECES[self.current_piece][rotation % len(PIECES[self.current_piece])]
        for dx in range(TEMPLATE_WIDTH):
            for dy in range(TEMPLATE_HEIGHT):
                if template[dy][dx] == 'O':
                    board_x_pos, board_y_pos = x + dx - 2, y + dy - 2
                    self.board[board_x_pos][board_y_pos] = 1

    def clearLines(self):
        """
        Removes completed lines from the board.
        Returns: int
            The number of lines removed.

        """
        lines_removed = 0
        y = 0  # start y at the bottom of the board
        while y < BOARD_HEIGHT:
            if self.isCompleteLine(y):
                # Remove the line and pull boxes down by one line.
                for pull_down_Y in range(y, BOARD_HEIGHT-1):
                    for x in range(BOARD_WIDTH):
                        self.board[x][pull_down_Y] = self.board[x][pull_down_Y + 1]
                # Set very top line to blank.
                for x in range(BOARD_WIDTH):
                    self.board[x][BOARD_HEIGHT] = BLANK
                lines_removed += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y += 1  # move on to check next row up
        return lines_removed

    def isCompleteLine(self, y):
        """
        Checks if the line at height y is complete.
        Args:
            y: int
                The height of the row to check.

        Returns: Boolean
            True if the row is complete.

        """
        for x in range(BOARD_WIDTH):
            if not self.board[y][x]: return False
        return True

    def convertToFeatures(self):
        """
        Converts the current board position and falling piece to a
        list of features.
        The features consist of:
            - 7 entries representing a 1 hot vector for the current piece.
            - BOARD_WIDTH entries representing the maximum height for each column.
            - BOARD_WIDTH - 1 entries representing the difference in heights between successive columns.
        Returns: torch tensor
            Torch tensor of the features described above. Values normalized to be between -1 and 1.

        """
        features = torch.zeros(len(PIECES) + 2 * BOARD_WIDTH - 1)

        # One hot vector for the current piece
        features[PIECES_IND[self.current_piece]] = 1.0

        # Maximum heights of each column
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT-1, -1, -1):
                if self.board[y][x]: break
            features[len(PIECES) + x] = y/BOARD_HEIGHT

        # Differences in heights between each column
        for x in range(BOARD_WIDTH-1):
            features[len(PIECES) + BOARD_WIDTH + x] = (features[len(PIECES) + x + 1] - features[len(PIECES) + x])/BOARD_HEIGHT

        return features.to(device)

    def countHoles(self):
        """
        Counts the number of transitions from filled to empty or vice
        versa in the rows and columns.
        Returns: tuple[int]
            A tuple (delta_r, delta_c) representing the number of transitions
            from filled to empty squares or vice versa across rows and columns respectively.

        """
        # Across rows:
        delta_r = 0
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH-1):
                if self.board[y][x] != self.board[y][x+1]:
                    delta_r += 1

        # Across columns:
        delta_c = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT-1):
                if self.board[y][x] != self.board[y+1][x]:
                    delta_c += 1

        return delta_r, delta_c