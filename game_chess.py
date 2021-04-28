import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import chess
import chess.svg
import numpy as np
import cv2


from MyChessFunction import *
from Calibration import *
from teachableMachine import *
from game_chess import *




# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")


    return board

def board_to_image(board):

    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")


    return board

def board2fen(chess_board):
    board_array = chess_board
    fen_line = ''
    count = 0
    for i in range(8):
        empty = 0
        for j in range(8):
            if board_array[i][j].isnumeric():
                empty+=1
            else:
                if empty != 0:
                    fen_line+= str(empty)+ str(board_array[i][j])
                    empty = 0
                else:
                    fen_line += str(board_array[i][j])
        if empty != 0:
            fen_line += str(empty)
        if count != 7:
            fen_line += str('/')
            count +=1
    fen_line += " w KQkq - 0 1"
    return fen_line

def save_positional(matrix):
    with open('outfile.txt', 'wb') as f:
        for line in matrix:
            np.savetxt(f, line, fmt='%i')

def get_positional_matrix():

    array = np.zeros((8,8))
    i = 0
    j = 0

    with open('outfile.txt') as f:
        for line in f:

            array[i][j] = (int(line.strip('\n')))

            j = j + 1

            if j == 8 :
                i = i + 1
                j = 0

            if i == 8 :
                return array


    print(array)
    return array

