import random
from game import Game, Move, Player
from collections import defaultdict
import numpy as np
import copy
import tqdm
import math

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

#PLAYER MINMAX WITH RL
class MyPlayerRLMinMax(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move_min_max(self, game: 'MyGame', current_player, alpha, beta, f, visited_states, depth, max_depth):
        '''Returns a tuple (reward_d, move). Where reward_d is a tuple (reward, depth) and move is a tuple (from_pos, slide)'''
        winner = game.check_winner()
        current_state = copy.deepcopy(game.current_state)   #Saves the current state
        current_board = copy.deepcopy(game.get_board())     #Saves the current board

        #Winner check
        if winner != -1:
            if winner == current_player:
                reward = 1
                d = -depth
            else:
                reward = -1
                d = depth
            
            if f != None: #For testing
                f.write(f"**********WINNER REACHED - PLAYER {winner}*************\n")
            return ((reward, d), None)
        
        #Hard cut off check
        if depth >= max_depth: #HARD CUT OFF
            if f != None: #For testing
                f.write(f"###################CUT OFF####################\n")
            return ((0, 0), None)

        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
        possible_moves = game.remove_symmetries(possible_moves, game.current_player_idx, current_board)
        
        #If it is the player turn
        if game.current_player_idx == current_player:
            #Sets max_value to (reward=-inf, depth=0)
            max_value = (-math.inf, 0)
            #Until there's still a move not yet considered
            while len(possible_moves) != 0:
                #Pick a random move from the possible ones
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                #Plays it
                game._Game__move(from_pos, slide, game.current_player_idx)
                #Updates the current_state
                game.current_state = game.board_to_state(game.get_board())

                if f != None: #For testing
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state) #Used as a key
                #Checks if the state has already been visited
                if (hashable_key, game.current_player_idx) in visited_states:
                    if f != None: #For testing
                        f.write("_________ALREADY VISITED STATE___________\n")
                    #If yes, just updates the reward
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    #Otherwise, next move
                    game.current_player_idx = (current_player+1)%2
                    reward, _ = self.make_move_min_max(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = current_player    #Restores the player index

                    visited_states[(hashable_key, game.current_player_idx)] = reward    #Adds state to visited_states

                game.current_state = copy.deepcopy(current_state)   #Restores the current state
                game._board = copy.deepcopy(current_board)  #Restores the current board
                
                #Updates max_value and optimal move
                if reward > max_value:
                    max_value = reward
                    opt_move = child_state
                
                #Updates alpha and checks for alpha beta pruning
                alpha = max(alpha, max_value[0])
                if alpha >= beta:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break
            
            return (max_value, opt_move)
        #If it is the opponent turn
        else:
            #Sets min_value to (reward=inf, depth=0)
            min_value = (math.inf, 0)
            #Until there's still a move not yet considered
            while len(possible_moves) != 0:
                #Pick a random move from the possible ones
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                #Plays it
                game._Game__move(from_pos, slide, game.current_player_idx)
                #Updates the current_state
                game.current_state = game.board_to_state(game.get_board())

                if f != None: #For testing
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)  #Used as a key
                #Checks if the state has already been visited
                if (hashable_key, game.current_player_idx) in visited_states:
                    if f != None: #For testing
                        f.write("_________ALREADY VISITED STATE___________\n")
                    #If yes, just updates the reward
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    #Otherwise, next move
                    game.current_player_idx = current_player
                    reward, _ = self.make_move_min_max(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = (current_player+1)%2  #Restores the player index

                    visited_states[(hashable_key, game.current_player_idx)] = reward    #Adds state to visited_states

                game.current_state = copy.deepcopy(current_state)   #Restores the current state
                game._board = copy.deepcopy(current_board)  #Restores the current board

                #Updates min_value and optimal move
                if reward < min_value:
                    min_value = reward
                    opt_move = child_state

                #Updates beta and checks for alpha beta pruning
                beta = min(beta, min_value[0])
                if beta <= alpha:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break

            return (min_value, opt_move)
        
    def make_move(self, game: 'MyGame', f, visited_states) -> tuple[tuple[int, int], Move]:
        '''Returns a tuple (from_pos, slide)'''
        equivalent_states = []  #List of all the possible equivalent states
        eq_state = []   #Stores the equivalent state found inside the Q-table
        current_state = copy.deepcopy(game.current_state)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        cs_in_q_table = (hashable_key_cs, game.current_player_idx) in game.qtables[game.current_player_idx]
        found = False
        num = 0         #Counter for the symmetry operations
        num_state = -1  #Stores the symmetry operation used

        #If the state is not present in the Q-table
        if not cs_in_q_table:
            for cs in game.rotate(game.get_board()):
                equivalent_states.append(cs)
            for cs in game.reflect(game.get_board()):
                equivalent_states.append(cs)

            #Checks if any equivalent state is inside the Q-table
            for es in equivalent_states:
                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in es)  #Used as a key

                if (hashable_key, game.current_player_idx) in game.qtables[game.current_player_idx]:
                    #Found an Equivalent State in Q-table
                    num_state = num
                    eq_state = copy.deepcopy(es)

                    max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
                    #Checks if the max Q-value is greater than 0.0
                    if max_Q > 0.0:
                        #Found an Equivalent State in Q-table, with a possible effective move
                        #print(f"FOUND EQUIVALENT STATE FOR:\n{current_state}\n ES: {es}")
                        found = True
                    
                        break

                num += 1
        #Otherwise
        else:
            #Found in Q-table
            hashable_key = hashable_key_cs
            max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
            #Checks if the max Q-value is greater than 0.0
            if max_Q > 0.0:
                #Found in Q-table, with a possible effective move
                #print(f"IN QTABLE\n{current_state}")
                found = True
        
        if found:
            #If the state or an equivalent one has been found inside the Q-table
            best_moves = []

            for i in game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)]:
                if game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)][i] == max_Q:
                    best_moves.append(i)

            from_pos, slide = best_moves[np.random.randint(len(best_moves))]

            #If it is an equivalent one, finds the equivalent move (from_pos, slide) for the current state
            if not cs_in_q_table:
                current_board = game.get_board()    #Saves the current board

                #Creates the board for the equivalent state
                game._board = game.state_to_board(eq_state)
                #Plays the optimal move
                game._Game__move(from_pos, slide, game.current_player_idx)

                #Operations to find the equivalent board for the next state
                if num_state == 0: #ROTATE 90 DEGREE
                    next_board = np.rot90(game.get_board(), k=3) #ROTATE 270 DEGREE
                elif num_state == 1: #ROTATE 180 DEGREE
                    next_board = np.rot90(game.get_board(), k=2) #ROTATE 180 DEGREE
                elif num_state == 2: #ROTATE 270 DEGREE
                    next_board = np.rot90(game.get_board(), k=1) #ROTATE 90 DEGREE
                elif num_state == 3: #REFLECT HORIZONTALLY
                    next_board = np.fliplr(game.get_board()) 
                elif num_state == 4: #REFLECT VERTICALLY
                    next_board = np.flipud(game.get_board()) 
                elif num_state == 5: #REFLECT DIAGONALLY
                    next_board = np.rot90(np.fliplr(game.get_board()))
                elif num_state == 6: #REFLECT ANTI-DIAGONALLY
                    next_board = np.rot90(np.flipud(game.get_board()))

                #Creates the board for the next state
                next_state = game.board_to_state(next_board)
                #Finds how to get to the next state from the current state
                moves = game.how_to_get_there(current_state, next_state)
                #Picks a random move from the possible ones
                from_pos, slide = moves[np.random.randint(len(moves))]

                game._board = copy.deepcopy(current_board)  #Restores the current board
        #Otherwise
        else:
            #Uses MinMax to find an optimal move
            #print(f"NEW STATE\n{current_state}")
            #Increasing the max_depth will result in better results but longer computational time
            _, opt_move = self.make_move_min_max(game, game.current_player_idx, -math.inf, math.inf, f, visited_states, 0, max_depth=1)
            from_pos, slide = opt_move

        return from_pos, slide

#PLAYER REINFORCEMENT LEARNING (QLEARNING)
class MyPlayerRL(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        '''Returns a tuple (from_pos, slide)'''
        equivalent_states = []  #List of all the possible equivalent states
        eq_state = []   #Stores the equivalent state found inside the Q-table
        current_state = copy.deepcopy(game.current_state)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        cs_in_q_table = (hashable_key_cs, game.current_player_idx) in game.qtables[game.current_player_idx]
        found = False
        num = 0         #Counter for the symmetry operations
        num_state = -1  #Stores the symmetry operation used

        #If the state is not present in the Q-table
        if not cs_in_q_table:
            for cs in game.rotate(game.get_board()):
                equivalent_states.append(cs)
            for cs in game.reflect(game.get_board()):
                equivalent_states.append(cs)

            #Checks if any equivalent state is inside the Q-table
            for es in equivalent_states:
                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in es)  #Used as a key

                if (hashable_key, game.current_player_idx) in game.qtables[game.current_player_idx]:
                    #Found an Equivalent State in Q-table
                    #print("EQUIVALENT STATE")
                    found = True
                    num_state = num
                    eq_state = copy.deepcopy(es)
                    max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
                    break

                num += 1

            if not found:
                #Picks a random move from the possible ones
                #print("NEW STATE")
                possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
                from_pos, slide = possible_moves[np.random.randint(len(possible_moves))]
        #Otherwise
        else:
            #Found in Q-table
            #print("IN QTABLE")
            found = True
            hashable_key = hashable_key_cs
            max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
        
        if found:
            #If the state or an equivalent one has been found inside the Q-table
            best_moves = []

            for i in game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)]:
                if game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)][i] == max_Q:
                    best_moves.append(i)

            from_pos, slide = best_moves[np.random.randint(len(best_moves))]

            #If it is an equivalent one, finds the equivalent move (from_pos, slide) for the current state
            if not cs_in_q_table:
                current_board = game.get_board()    #Saves the current board

                #Creates the board for the equivalent state
                game._board = game.state_to_board(eq_state) 
                #Plays the optimal move
                game._Game__move(from_pos, slide, game.current_player_idx)  

                #Operations to find the equivalent board for the next state
                if num_state == 0: #ROTATE 90 DEGREE
                    next_board = np.rot90(game.get_board(), k=3) #ROTATE 270 DEGREE
                elif num_state == 1: #ROTATE 180 DEGREE
                    next_board = np.rot90(game.get_board(), k=2) #ROTATE 180 DEGREE
                elif num_state == 2: #ROTATE 270 DEGREE
                    next_board = np.rot90(game.get_board(), k=1) #ROTATE 90 DEGREE
                elif num_state == 3: #REFLECT HORIZONTALLY
                    next_board = np.fliplr(game.get_board()) 
                elif num_state == 4: #REFLECT VERTICALLY
                    next_board = np.flipud(game.get_board()) 
                elif num_state == 5: #REFLECT DIAGONALLY
                    next_board = np.rot90(np.fliplr(game.get_board()))
                elif num_state == 6: #REFLECT ANTI-DIAGONALLY
                    next_board = np.rot90(np.flipud(game.get_board()))

                #Creates the board for the next state
                next_state = game.board_to_state(next_board)
                #Finds how to get to the next state from the current state
                moves = game.how_to_get_there(current_state, next_state)
                #Picks a random move from the possible ones
                from_pos, slide = moves[np.random.randint(len(moves))]

                game._board = copy.deepcopy(current_board)  #Restores the current board

        return from_pos, slide

#PLAYER MINMAX
class MyPlayerMinMax(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame', current_player, alpha, beta, f, visited_states, depth, max_depth):
        '''Returns a tuple (reward_d, move). Where reward_d is a tuple (reward, depth) and move is a tuple (from_pos, slide)'''
        winner = game.check_winner()
        current_state = copy.deepcopy(game.current_state)   #Saves the current state
        current_board = copy.deepcopy(game.get_board())     #Saves the current board

        #Winner check
        if winner != -1:
            if winner == current_player:
                reward = 1
                d = -depth
            else:
                reward = -1
                d = depth
            
            if f != None: #For testing
                f.write(f"**********WINNER REACHED - PLAYER {winner}*************\n")
            return ((reward, d), None)
        
        #Hard cut off check
        if depth >= max_depth: #HARD CUT OFF
            if f != None: #For testing
                f.write(f"###################CUT OFF####################\n")
            return ((0, 0), None)

        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
        possible_moves = game.remove_symmetries(possible_moves, game.current_player_idx, current_board)
        
        #If it is the player turn
        if game.current_player_idx == current_player:
            #Sets max_value to (reward=-inf, depth=0)
            max_value = (-math.inf, 0)
            #Until there's still a move not yet considered
            while len(possible_moves) != 0:
                #Pick a random move from the possible ones
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                #Plays it
                game._Game__move(from_pos, slide, game.current_player_idx)
                #Updates the current_state
                game.current_state = game.board_to_state(game.get_board())

                if f != None: #For testing
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state) #Used as a key
                #Checks if the state has already been visited
                if (hashable_key, game.current_player_idx) in visited_states:
                    if f != None: #For testing
                        f.write("_________ALREADY VISITED STATE___________\n")
                    #If yes, just updates the reward
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    #Otherwise, next move
                    game.current_player_idx = (current_player+1)%2
                    reward, _ = self.make_move(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = current_player    #Restores the player index

                    visited_states[(hashable_key, game.current_player_idx)] = reward    #Adds state to visited_states

                game.current_state = copy.deepcopy(current_state)   #Restores the current state
                game._board = copy.deepcopy(current_board)  #Restores the current board
                
                #Updates max_value and optimal move
                if reward > max_value:
                    max_value = reward
                    opt_move = child_state
                
                #Updates alpha and checks for alpha beta pruning
                alpha = max(alpha, max_value[0])
                if alpha >= beta:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break
            
            return (max_value, opt_move)
        #If it is the opponent turn
        else:
            #Sets min_value to (reward=inf, depth=0)
            min_value = (math.inf, 0)
            #Until there's still a move not yet considered
            while len(possible_moves) != 0:
                #Pick a random move from the possible ones
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                #Plays it
                game._Game__move(from_pos, slide, game.current_player_idx)
                #Updates the current_state
                game.current_state = game.board_to_state(game.get_board())

                if f != None: #For testing
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)  #Used as a key
                #Checks if the state has already been visited
                if (hashable_key, game.current_player_idx) in visited_states:
                    if f != None: #For testing
                        f.write("_________ALREADY VISITED STATE___________\n")
                    #If yes, just updates the reward
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    #Otherwise, next move
                    game.current_player_idx = current_player
                    reward, _ = self.make_move(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = (current_player+1)%2  #Restores the player index

                    visited_states[(hashable_key, game.current_player_idx)] = reward    #Adds state to visited_states

                game.current_state = copy.deepcopy(current_state)   #Restores the current state
                game._board = copy.deepcopy(current_board)  #Restores the current board

                #Updates min_value and optimal move
                if reward < min_value:
                    min_value = reward
                    opt_move = child_state

                #Updates beta and checks for alpha beta pruning
                beta = min(beta, min_value[0])
                if beta <= alpha:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break

            return (min_value, opt_move)


#GAME
class MyGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.qtables = {}   #A Q-table for each RL player
        #What is a state? (list of pieces (x,y) taken by player1, list of pieces (x,y) taken by player2)
        self.current_state = ([], [])   #Keeps track of the current state
        self.next_state = ([], [])  #Keeps track of the next state

    def board_to_state(self, board: np.ndarray):
        '''Given a board, it returns the corresponding state'''
        zero_set = []
        one_set  = []

        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if board[row][column] == 0:
                    zero_set.append((column, row))
                elif board[row][column] == 1:
                    one_set.append((column, row))

        return (sorted(zero_set), sorted(one_set))
    
    def state_to_board(self, state):
        '''Given a state, it returns the corresponding board'''
        board = np.ones((5, 5), dtype=np.uint8) * -1

        for s in state[0]:
            board[s[1]][s[0]] = 0

        for s in state[1]:
            board[s[1]][s[0]] = 1

        return board
    
    def remove_symmetries(self, moves, player_id, current_board):
        '''Given a board, the player_id, and the list of possible moves. Removes the symmetric moves and returns the new list'''
        new_possible_moves = []
        states = [] #Keeps track of all the new states found, considering symmetries

        #For each possible move
        for s in moves:
            self._Game__move(s[0], s[1], player_id)

            new_state = self.board_to_state(self.get_board())

            #If the state or one of its equivalent states has already been visited
            if new_state in states:
                continue
            
            #Otherwise updates the list
            rotated_states = self.rotate(self.get_board())
            states.append(rotated_states)

            reflected_states = self.reflect(self.get_board())
            states.append(reflected_states)

            states.append(new_state)
            new_possible_moves.append(s)

            #Restores the current board
            self._board = copy.deepcopy(current_board)

        return new_possible_moves
    
    def rotate(self, board):
        '''Starting from a board, creates a board for each rotation, returns a list containing all the rotated boards '''
        rotated_states = []

        board1 = np.rot90(board)    #ROTATE 90 DEGREE
        rotated_states.append(self.board_to_state(board1))
        board2 = np.rot90(board, k=2) #ROTATE 180 DEGREE
        rotated_states.append(self.board_to_state(board2))
        board3 = np.rot90(board, k=3)   #ROTATE 270 DEGREE
        rotated_states.append(self.board_to_state(board3))
                    
        return rotated_states

    def reflect(self, board):
        '''Starting from a board, creates a board for each reflection, returns a list containing all the reflected boards '''
        reflected_states = []

        board1 = np.fliplr(board)   #REFLECT HORIZONTALLY
        reflected_states.append(self.board_to_state(board1))
        board2 = np.flipud(board)   #REFLECT VERTICALLY
        reflected_states.append(self.board_to_state(board2))
        board3 = np.rot90(np.fliplr(board))    #REFLECT DIAGONALLY
        reflected_states.append(self.board_to_state(board3))
        board4 = np.rot90(np.flipud(board))    #REFLECT ANTI-DIAGONALLY
        reflected_states.append(self.board_to_state(board4))
                    
        return reflected_states

    def get_possible_moves(self, state, player_id):
        '''Starting from a state and a player id, returns all the possible moves of that player'''
        #All the border pieces
        possible_from_pos = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)]
        possible_actions = []
        possible_moves = []

        #Removes pieces of the opponent player
        if player_id == 0:
            possible_from_pos = [p for p in possible_from_pos if p not in state[1]]
        elif player_id == 1:
            possible_from_pos = [p for p in possible_from_pos if p not in state[0]]

        #For each from_pos, find the possible moves
        for p in possible_from_pos:
            possible_actions = []

            if p != (0,0) and p != (1,0) and p != (2,0) and p != (3,0) and p != (4,0):
                possible_actions.append(Move.TOP)
            
            if p != (4,0) and p != (4,1) and p != (4,2) and p != (4,3) and p != (4,4):
                possible_actions.append(Move.RIGHT)

            if p != (0,4) and p != (1,4) and p != (2,4) and p != (3,4) and p != (4,4):
                possible_actions.append(Move.BOTTOM)

            if p != (0,4) and p != (0,3) and p != (0,2) and p != (0,1) and p != (0,0):
                possible_actions.append(Move.LEFT)

            for s in possible_actions:
                possible_moves.append((p,s))

        return possible_moves

    def how_to_get_there(self, starting_state, final_state):
        '''Starting from a starting_state,
           returns the list of all the possible moves (from_pos, slide) that brings to the final_state'''
        moves = []
        current_board = copy.deepcopy(self.get_board())     #Saves the actual board
        board = self.state_to_board(starting_state)  #Creates and saves the board corresponding to the starting_state

        possible_moves = self.get_possible_moves(starting_state, self.current_player_idx)

        for p in possible_moves:
            self._board = copy.deepcopy(board)      #Restores the starting_state board
            self._Game__move(p[0], p[1], self.current_player_idx)

            state = self.board_to_state(self.get_board())

            if state[0] == final_state[0] and state[1] == final_state[1]:
                moves.append((p[0], p[1]))

        self._board = copy.deepcopy(current_board)          #Restores the actual board

        return moves

    def initialize_qtable(self, possible_moves):
        moves_values = {}

        for j in possible_moves:
            moves_values[j] = 0.0

        return moves_values
    
    def update_Q_table(self, qtable_index, alpha, gamma, current_state, from_pos, slide, reward, next_state) -> None:
        '''Updates the Q-table'''
        possible_moves_cs = self.get_possible_moves(copy.deepcopy(current_state), self.current_player_idx)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)    #Used as index for the q-table
        #If not available, adds the current_state to the Q-table
        if (hashable_key_cs, self.current_player_idx) not in self.qtables[qtable_index]:
            self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)] = self.initialize_qtable(possible_moves_cs)

        next_player = (self.current_player_idx+1)%2
        hashable_key_ns = tuple(tuple(inner_tuple) for inner_tuple in next_state)   #Used as index for the q-table
        #If the next_state is in the Q-table, finds the max Q-value for the next state
        if (hashable_key_ns, next_player) in self.qtables[qtable_index]:
            max_next_Q = max(self.qtables[qtable_index][(hashable_key_ns, next_player)].values())
        else:
        #Otherwise, it sets the value to 0.0
            max_next_Q = 0.0

        #Update Q-table
        self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] = (1 - alpha) * self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] + alpha * (reward + gamma * max_next_Q)

    def train_agent(self) -> None:
        '''Trains the agent'''
        print("TRAINING")
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        for _ in tqdm.tqdm(range(10_000)):
            self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD
            self.current_player_idx = 1 #RESET PLAYER INDEX
            self.playGame(player1, player2, training=True, testing=False)

    def playGame(self, player1: "Player", player2: "Player", training: bool, testing: bool) -> int:
        '''Plays the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        visited_states = {} #Keeps track of already visited states
        f = None

        if testing:
            f = open("quixo/play.txt", "w+")

        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False

            #Updates the current state
            self.current_state = self.board_to_state(self.get_board())

            while not ok:
                if isinstance(players[self.current_player_idx], MyPlayerMinMax):
                    #Increasing the max_depth will result in better results but longer computational time
                    _, opt_move = players[self.current_player_idx].make_move(self, self.current_player_idx, -math.inf, math.inf, f, visited_states, 0, max_depth=2)
                elif isinstance(players[self.current_player_idx], MyPlayerRLMinMax):
                    opt_move = players[self.current_player_idx].make_move(self, f, visited_states)
                else:
                    opt_move = players[self.current_player_idx].make_move(self)
                
                if testing:
                    print(f"TESTING {opt_move} By PLAYER {self.current_player_idx}")
                
                ok = self._Game__move(opt_move[0], opt_move[1], self.current_player_idx)

            #Updates the next state
            self.next_state = self.board_to_state(self.get_board())

            winner = self.check_winner()

            if training:
                #For each RL player, updates their Q-table
                for r in self.qtables:
                    if winner == -1:
                        reward = 0
                    elif winner == r:
                        reward = 1
                    else:
                        reward = -1

                    self.update_Q_table(r, 0.1, 0.9, self.current_state, opt_move[0], opt_move[1], reward, self.next_state)

            #Updates the current state
            self.current_state = copy.deepcopy(self.next_state)

            if testing:
                print(f"PLAYER {self.current_player_idx} DID OPT_MOVE: {opt_move}")
                print(f"CURRENT STATE: {self.current_state}")
                print(f"BOARD:\n {self.get_board()}")

        if testing:
            f.close()
            
        return winner


if __name__ == '__main__':
    num_of_matches = 100    #Number of matches to play
    num_wins_0 = 0          #Counter for player1 wins
    num_wins_1 = 0          #Counter for player2 wins

    #OTHER ATTEMPTS:
    #MyPlayerRL uses Reinforcement Learning (Q-learning)    #NOT WORKING GREAT
    #MyPlayerRLMinMax uses a mix of RL and MinMax   #NOT AN IMPROVEMENT COMPARED TO MinMax
    
    player1 = MyPlayerMinMax()
    player2 = RandomPlayer()
    
    players = [player1, player2]

    g = MyGame()

    #Initializes the Q-table for each RL player
    num_rl_players = 0
    for player_index in range(len(players)):
        if isinstance(players[player_index], MyPlayerRLMinMax) or isinstance(players[player_index], MyPlayerRL):
            g.qtables[player_index] = defaultdict(list)
            num_rl_players += 1
    
    #If there's at least 1 RL player, let the training begin
    if num_rl_players > 0:
        g.train_agent()

    print(f"PLAYING {num_of_matches} MATCHES")
    for _ in tqdm.tqdm(range(num_of_matches)):
        g._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD
        g.current_player_idx = 1 #RESET PLAYER INDEX

        #With testing=True prints the game in play.txt if one of the users uses MinMax
        #training=False is to keep, because =True is used only by the train_agent function in case of an RL player
        winner = g.playGame(player1, player2, training=False, testing=False)
        
        if winner == 0:
            num_wins_0 += 1
        elif winner == 1:
            num_wins_1 += 1
    
    print(f"WINS PLAYER 0: {num_wins_0}, WINS PLAYER 1: {num_wins_1}")
