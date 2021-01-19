import random

# Step 1: Write a function that can print out a board.
# Set up your board as a list, where each index 1-9 corresponds with a number on a number pad,
# so you get a 3 by 3 board representation.

def print_board(lst):
    print(lst[6], "|", lst[7], "|", lst[8])
    print(" -------")
    print(lst[3], "|", lst[4], "|", lst[5])
    print(" -------")
    print(lst[0], "|", lst[1], "|", lst[2])


# test_board = ['#', 'X', 'O', 'X', 'O', 'X', 'O', 'X', 'O', 'X']
#
# print_board(test_board)


# Step 2: Write a function that can take in a player input and assign their marker as 'X' or 'O'.
# Think about using while loops to continually ask until you get a correct answer.

def player_input():
    marker = ''
    while not (marker == 'X' or marker == 'O'):
        marker = input('Player 1: Do you want to be X or O? ').upper()
    if marker == 'X':
        return 'X', 'O'
    else:
        return 'O', 'X'

#
# player_input()


# Step 3: Write a function that takes in the board list object,
# a marker ('X' or 'O'), and a desired position (number 1-9) and assigns it to the board.

def place_marker(board, marker, position):
    board[position] = marker

#
# place_marker(test_board, '$', 8)
# print_board(test_board)


# Step 4: Write a function that takes in a board and checks to see if someone has won.


def win_check(board, mark):
    return ((board[7] == mark and board[8] == mark and board[9] == mark) or  # top
            (board[4] == mark and board[5] == mark and board[6] == mark) or  # middle
            (board[1] == mark and board[2] == mark and board[3] == mark) or  # bottom
            (board[7] == mark and board[4] == mark and board[1] == mark) or  # left
            (board[8] == mark and board[5] == mark and board[2] == mark) or  # middle
            (board[9] == mark and board[6] == mark and board[3] == mark) or  # right
            (board[7] == mark and board[5] == mark and board[3] == mark) or  # diagonal
            (board[9] == mark and board[5] == mark and board[1] == mark))  # diagonal


# win_check(test_board, 'X')


# Step 5: Write a function that uses the random module to randomly decide which player goes
# first. You may want to lookup random.randint() Return a string of which player went first.

def choose_first():
    if random.randint(0, 1) == 0:
        return 'Player 2'
    else:
        return 'Player 1'


# Step 6: Write a function that returns a boolean indicating whether a space on the board is freely available.

def space_check(board, position):
    return board[position] == ' '


# Step 7: Write a function that checks if the board is full and returns a boolean value.
# True if full, False otherwise.

def full_board_check(board):
    for i in range(1, 10):
        if space_check(board, i):
            return False
    return True


# Step 8: Write a function that asks for a player's next position (as a number 1-9)
# and then uses the function from step 6 to check if it's a free position.
# If it is, then return the position for later use.

def player_choice(board):
    position = 0

    while position not in [1, 2, 3, 4, 5, 6, 7, 8, 9] or not space_check(board, position):
        position = int(input('Choose your next position: (1-9) '))

    return position


# Step 9: Write a function that asks the player if they
# want to play again and returns a boolean True if they do want to play again.
def replay():
    return input('Do you want to play again? Enter Yes or No: ')


print('Welcome to Tic Tac Toe!')

while True:
    board = [' '] * 10
    player1_mark, player2_mark = player_input()
    turn = choose_first()
    print(turn + ' will go first.')

    play_game = input('Are you ready to play? Enter Yes or No.')

    if play_game[0] == 'Y':
        game_on = True
    else:
        game_on = False

    while game_on:
        if turn == 'Player 1':

            print_board(board)
            position = player_choice(board)
            place_marker(board, player1_mark, position)

            if win_check(board, player1_mark):
                print_board(board)
                print('Player 1 wins!')
                game_on = False
            else:
                if full_board_check(board):
                    print_board(board)
                    print('Draw!')
                    break
                else:
                    turn = 'Player 2'

        else:

            print_board(board)
            position = player_choice(board)
            place_marker(board, player2_mark, position)

            if win_check(board, player2_mark):
                print_board(board)
                print('Player 2 wins!')
                game_on = False
            else:
                if full_board_check(board):
                    print_board(board)
                    print('Draw!')
                    break
                else:
                    turn = 'Player 1'

        if not replay():
            break


