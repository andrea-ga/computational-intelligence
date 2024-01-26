import random
from game import Game, Move, Player
from collections import defaultdict
import numpy as np
import copy
import tqdm
import pickle
import math

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

#MINMAX WITH RL
class MyPlayerRLMinMax(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move_min_max(self, game: 'RLMinMaxGame', current_player, alpha, beta, f, visited_states, depth, max_depth):
        winner = game.check_winner()
        current_state = copy.deepcopy(game.current_state)
        current_board = copy.deepcopy(game.get_board())

        if winner != -1:
            if winner == current_player:
                reward = 1
                d = -depth
            else:
                reward = -1
                d = depth
            
            if f != None:
                f.write(f"**********WINNER REACHED - PLAYER {winner}*************\n")
            return ((reward, d), None)
        
        if depth >= max_depth: #HARD CUT OFF
            if f != None:
                f.write(f"###################CUT OFF####################\n")
            return ((0, 0), None)

        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
        possible_moves = game.remove_symmetries(possible_moves, game.current_player_idx, current_board)
        
        if game.current_player_idx == current_player:
            max_value = (-math.inf, 0)
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game._Game__move(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                if f != None:
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)
                if (hashable_key, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    if f != None:
                        f.write("_________ALREADY VISITED STATE___________\n")
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    game.current_player_idx = (current_player+1)%2
                    reward, _ = self.make_move_min_max(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = current_player

                    visited_states[(hashable_key, game.current_player_idx)] = reward

                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if reward > max_value:
                    max_value = reward
                    opt_move = child_state
                
                alpha = max(alpha, max_value[0])
                if alpha >= beta:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break
            
            return (max_value, opt_move)
        else:
            min_value = (math.inf, 0)
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game._Game__move(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                if f != None:
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)
                if (hashable_key, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    if f != None:
                        f.write("_________ALREADY VISITED STATE___________\n")
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    game.current_player_idx = current_player
                    reward, _ = self.make_move_min_max(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = (current_player+1)%2

                    visited_states[(hashable_key, game.current_player_idx)] = reward

                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if reward < min_value:
                    min_value = reward
                    opt_move = child_state

                beta = min(beta, min_value[0])
                if beta <= alpha:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break

            return (min_value, opt_move)
        
    def make_move(self, game: 'RLMinMaxGame', f, visited_states) -> tuple[tuple[int, int], Move]:
        equivalent_states = []
        eq_state = []
        current_state = copy.deepcopy(game.current_state)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        cs_in_q_table = (hashable_key_cs, game.current_player_idx) in game.qtables[game.current_player_idx]
        found = False
        num = 0
        num_state = -1

        if not cs_in_q_table:
            for cs in game.rotate(game.get_board()):
                equivalent_states.append(cs)
            for cs in game.reflect(game.get_board()):
                equivalent_states.append(cs)

            for es in equivalent_states:
                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in es)

                if (hashable_key, game.current_player_idx) in game.qtables[game.current_player_idx]:
                    num_state = num
                    eq_state = copy.deepcopy(es)

                    max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
                    if max_Q > 0.0:
                        found = True
                        print(f"FOUND EQUIVALENT STATE FOR:\n{current_state}\n ES: {es}")
                    
                        break

                num += 1
        else:
            hashable_key = hashable_key_cs
            max_Q = max(game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)].values())
            if max_Q > 0.0:
                found = True
                print(f"IN QTABLE\n{current_state}")
        
        if found:
            best_moves = []

            for i in game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)]:
                if game.qtables[game.current_player_idx][(hashable_key, game.current_player_idx)][i] == max_Q:
                    best_moves.append(i)

            from_pos, slide = best_moves[np.random.randint(len(best_moves))]

            if not cs_in_q_table:
                current_board = game.get_board()

                game._board = game.set_to_matrix(eq_state)
                game._Game__move(from_pos, slide, game.current_player_idx)

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

                next_state = game.matrix_to_set(next_board)

                moves = game.how_to_get_there(current_state, next_state)

                from_pos, slide = moves[np.random.randint(len(moves))]

                game._board = copy.deepcopy(current_board)
        else:
            print(f"NEW STATE\n{current_state}")
            _, opt_move = self.make_move_min_max(game, game.current_player_idx, -math.inf, math.inf, f, visited_states, 0, 1)
            from_pos, slide = opt_move

        return from_pos, slide

class RLMinMaxGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.current_state = ([], [])
        self.qtables = {}

    def matrix_to_set(self, board: np.ndarray):
        zero_set = []
        one_set  = []

        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if board[row][column] == 0:
                    zero_set.append((column, row))
                elif board[row][column] == 1:
                    one_set.append((column, row))

        return (sorted(zero_set), sorted(one_set))
    
    def set_to_matrix(self, state):
        board = np.ones((5, 5), dtype=np.uint8) * -1

        for s in state[0]:
            board[s[1]][s[0]] = 0

        for s in state[1]:
            board[s[1]][s[0]] = 1

        return board
    
    def remove_symmetries(self, moves, player_id, current_board):
        new_possible_moves = []
        states = []

        for s in moves:
            self._Game__move(s[0], s[1], player_id)

            new_state = self.matrix_to_set(self.get_board())

            if new_state in states:
                continue

            rotated_states = self.rotate(self.get_board())
            states.append(rotated_states)

            reflected_states = self.reflect(self.get_board())
            states.append(reflected_states)

            states.append(new_state)
            new_possible_moves.append(s)

            self._board = copy.deepcopy(current_board)

        return new_possible_moves
    
    def rotate(self, board):
        rotated_states = []

        board1 = np.rot90(board)    #ROTATE 90 DEGREE
        rotated_states.append(self.matrix_to_set(board1))
        board2 = np.rot90(board, k=2) #ROTATE 180 DEGREE
        rotated_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(board, k=3)   #ROTATE 270 DEGREE
        rotated_states.append(self.matrix_to_set(board3))
                    
        return rotated_states

    def reflect(self, board):
        reflected_states = []

        board1 = np.fliplr(board)   #REFLECT HORIZONTALLY
        reflected_states.append(self.matrix_to_set(board1))
        board2 = np.flipud(board)   #REFLECT VERTICALLY
        reflected_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(np.fliplr(board))    #REFLECT DIAGONALLY
        reflected_states.append(self.matrix_to_set(board3))
        board4 = np.rot90(np.flipud(board))    #REFLECT ANTI-DIAGONALLY
        reflected_states.append(self.matrix_to_set(board4))
                    
        return reflected_states

    def get_possible_moves(self, state, player_id):
        possible_from_pos = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)]
        possible_actions = []
        possible_moves = []
        
        if player_id == 0:
            possible_from_pos = [p for p in possible_from_pos if p not in state[1]]
        elif player_id == 1:
            possible_from_pos = [p for p in possible_from_pos if p not in state[0]]

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

    def how_to_get_there(self, current_state, next_state):
        moves = []
        current_board = copy.deepcopy(self.get_board())
        board = self.set_to_matrix(current_state)

        possible_moves = self.get_possible_moves(current_state, self.current_player_idx)

        for p in possible_moves:
            self._board = copy.deepcopy(board)
            self._Game__move(p[0], p[1], self.current_player_idx)

            state = self.matrix_to_set(self.get_board())

            if state[0] == next_state[0] and state[1] == next_state[1]:
                moves.append((p[0], p[1]))

        self._board = copy.deepcopy(current_board)

        return moves

    def initialize_qtable(self, possible_moves):
        moves_values = {}

        for j in possible_moves:
            moves_values[j] = 0.0

        return moves_values
    
    def update_Q_table(self, qtable_index, alpha, gamma, current_state, from_pos, slide, reward, next_state) -> None:
        possible_moves_cs = self.get_possible_moves(copy.deepcopy(current_state), self.current_player_idx)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        if (hashable_key_cs, self.current_player_idx) not in self.qtables[qtable_index]:
            self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)] = self.initialize_qtable(possible_moves_cs)

        next_player = (self.current_player_idx+1)%2
        hashable_key_ns = tuple(tuple(inner_tuple) for inner_tuple in next_state)
        if (hashable_key_ns, next_player) in self.qtables[qtable_index]:
            max_next_Q = max(self.qtables[qtable_index][(hashable_key_ns, next_player)].values())
        else:
            max_next_Q = 0.0

        #TIME DIFFERENCE
        self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] = (1 - alpha) * self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] + alpha * (reward + gamma * max_next_Q)
        #MONTE CARLO
        #self.qtables[qtable_index][current_state][m] = self.qtables[qtable_index][current_state][m] + alpha * reward

    def train_agent(self) -> None:
        print("TRAINING")
        for _ in tqdm.tqdm(range(1_000)):
            self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD
            self.current_player_idx = 1 #RESET PLAYER INDEX
            player1 = RandomPlayer()
            player2 = RandomPlayer()
            self.playRL(True, player1, player2)

    def playRL(self, training: bool, player1: "Player", player2: "Player") -> int:
        '''Play the game. Update qtable. Returns the winning player'''
        players = [player1, player2]
        winner = -1

        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            from_pos = -1
            slide = -1

            self.current_state = self.matrix_to_set(self.get_board())

            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
                
            self.next_state = self.matrix_to_set(self.get_board())

            winner = self.check_winner()

            if training:
                for r in self.qtables:
                    if winner == -1:
                        reward = 0
                    elif winner == r:
                        reward = 1
                    else:
                        reward = -1

                    self.update_Q_table(r, 0.1, 0.9, self.current_state, from_pos, slide, reward, self.next_state)
            
        return winner

    def playRLMinMax(self, player1: "Player", player2: "Player", testing: bool) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        visited_states = {}
        f = None

        if testing:
            f = open("quixo/play.txt", "w+")
        
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            self.current_state = self.matrix_to_set(self.get_board())

            while not ok:
                if isinstance(players[self.current_player_idx], MyPlayerRLMinMax):
                    opt_move = players[self.current_player_idx].make_move(self, f, visited_states)
                elif isinstance(players[self.current_player_idx], RandomPlayer):
                    opt_move = players[self.current_player_idx].make_move(self)
                
                if testing:
                    print(f"TESTING {opt_move} By PLAYER {self.current_player_idx}")
                
                ok = self._Game__move(opt_move[0], opt_move[1], self.current_player_idx)

            self.current_state = self.matrix_to_set(self.get_board())

            winner = self.check_winner()

            if testing:
                print(f"PLAYER {self.current_player_idx} DID OPT_MOVE: {opt_move}")
                print(f"CURRENT STATE: {self.current_state}")
                print(f"BOARD:\n {self.get_board()}")

        if testing:
            f.close()
            
        return winner


#MINMAX
class MyPlayerMinMax(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MinMaxGame', current_player, alpha, beta, f, visited_states, depth, max_depth):
        winner = game.check_winner()
        current_state = copy.deepcopy(game.current_state)
        current_board = copy.deepcopy(game.get_board())

        if winner != -1:
            if winner == current_player:
                reward = 1
                d = -depth
            else:
                reward = -1
                d = depth
            
            if f != None:
                f.write(f"**********WINNER REACHED - PLAYER {winner}*************\n")
            return ((reward, d), None)
        
        if depth >= max_depth: #HARD CUT OFF
            if f != None:
                f.write(f"###################CUT OFF####################\n")
            return ((0, 0), None)

        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
        possible_moves = game.remove_symmetries(possible_moves, game.current_player_idx, current_board)
        
        if game.current_player_idx == current_player:
            max_value = (-math.inf, 0)
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game._Game__move(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                if f != None:
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)
                if (hashable_key, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    if f != None:
                        f.write("_________ALREADY VISITED STATE___________\n")
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    game.current_player_idx = (current_player+1)%2
                    reward, _ = self.make_move(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = current_player

                    visited_states[(hashable_key, game.current_player_idx)] = reward

                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if reward > max_value:
                    max_value = reward
                    opt_move = child_state
                
                alpha = max(alpha, max_value[0])
                if alpha >= beta:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break
            
            return (max_value, opt_move)
        else:
            min_value = (math.inf, 0)
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game._Game__move(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                if f != None:
                    f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                    f.write(f"STATE: {game.current_state}\n")
                    f.write(f"BOARD:\n{game.get_board()}\n")

                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in game.current_state)
                if (hashable_key, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    if f != None:
                        f.write("_________ALREADY VISITED STATE___________\n")
                    reward = visited_states[(hashable_key, game.current_player_idx)]
                else:
                    game.current_player_idx = current_player
                    reward, _ = self.make_move(game, current_player, alpha, beta, f, visited_states, depth+1, max_depth)

                    game.current_player_idx = (current_player+1)%2

                    visited_states[(hashable_key, game.current_player_idx)] = reward

                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if reward < min_value:
                    min_value = reward
                    opt_move = child_state

                beta = min(beta, min_value[0])
                if beta <= alpha:
                    if f != None:
                        f.write("---------------PRUNING-----------------\n")
                    break

            return (min_value, opt_move)

class MinMaxGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.current_state = ([], [])
        self.qtable = defaultdict(list)

    def matrix_to_set(self, board: np.ndarray):
        zero_set = []
        one_set  = []

        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if board[row][column] == 0:
                    zero_set.append((column, row))
                elif board[row][column] == 1:
                    one_set.append((column, row))

        return (sorted(zero_set), sorted(one_set))
    
    def remove_symmetries(self, moves, player_id, current_board):
        new_possible_moves = []
        states = []

        for s in moves:
            self._Game__move(s[0], s[1], player_id)

            new_state = self.matrix_to_set(self.get_board())

            if new_state in states:
                continue

            rotated_states = self.rotate(self.get_board())
            states.append(rotated_states)

            reflected_states = self.reflect(self.get_board())
            states.append(reflected_states)

            states.append(new_state)
            new_possible_moves.append(s)

            self._board = copy.deepcopy(current_board)

        return new_possible_moves
    
    def rotate(self, board):
        rotated_states = []

        board1 = np.rot90(board)    #ROTATE 90 DEGREE
        rotated_states.append(self.matrix_to_set(board1))
        board2 = np.rot90(board, k=2) #ROTATE 180 DEGREE
        rotated_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(board, k=3)   #ROTATE 270 DEGREE
        rotated_states.append(self.matrix_to_set(board3))
                    
        return rotated_states

    def reflect(self, board):
        reflected_states = []

        board1 = np.fliplr(board)   #REFLECT HORIZONTALLY
        reflected_states.append(self.matrix_to_set(board1))
        board2 = np.flipud(board)   #REFLECT VERTICALLY
        reflected_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(np.fliplr(board))    #REFLECT DIAGONALLY
        reflected_states.append(self.matrix_to_set(board3))
        board4 = np.rot90(np.flipud(board))    #REFLECT ANTI-DIAGONALLY
        reflected_states.append(self.matrix_to_set(board4))
                    
        return reflected_states

    def get_possible_moves(self, state, player_id):
        possible_from_pos = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)]
        possible_actions = []
        possible_moves = []
        
        if player_id == 0:
            possible_from_pos = [p for p in possible_from_pos if p not in state[1]]
        elif player_id == 1:
            possible_from_pos = [p for p in possible_from_pos if p not in state[0]]

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

    def playMinMax(self, player1: "Player", player2: "Player", testing: bool) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        visited_states = {}
        f = None

        if testing:
            f = open("quixo/play.txt", "w+")
        
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            self.current_state = self.matrix_to_set(self.get_board())

            while not ok:
                if isinstance(players[self.current_player_idx], MyPlayerMinMax):
                    _, opt_move = players[self.current_player_idx].make_move(self, self.current_player_idx, -math.inf, math.inf, f, visited_states, 0, 1)
                elif isinstance(players[self.current_player_idx], RandomPlayer):
                    opt_move = players[self.current_player_idx].make_move(self)
                
                if testing:
                    print(f"TESTING {opt_move} By PLAYER {self.current_player_idx}")
                
                ok = self._Game__move(opt_move[0], opt_move[1], self.current_player_idx)

            self.current_state = self.matrix_to_set(self.get_board())

            winner = self.check_winner()

            if testing:
                print(f"PLAYER {self.current_player_idx} DID OPT_MOVE: {opt_move}")
                print(f"CURRENT STATE: {self.current_state}")
                print(f"BOARD:\n {self.get_board()}")

        if testing:
            f.close()
            
        return winner


#REINFORCEMENT LEARNING (QLEARNING)
class MyPlayerRL(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'RLGame') -> tuple[tuple[int, int], Move]:
        equivalent_states = []
        eq_state = []
        current_state = copy.deepcopy(game.current_state)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        cs_in_q_table = (hashable_key_cs, game.current_player_idx) in game.qtable
        found = False
        num = 0
        num_state = -1

        if not cs_in_q_table:
            for cs in game.rotate(game.get_board()):
                equivalent_states.append(cs)
            for cs in game.reflect(game.get_board()):
                equivalent_states.append(cs)

            for es in equivalent_states:
                #print("EQUIVALENT STATE")
                hashable_key = tuple(tuple(inner_tuple) for inner_tuple in es)

                if (hashable_key, game.current_player_idx) in game.qtable:
                    found = True
                    num_state = num
                    eq_state = copy.deepcopy(es)
                    max_Q = max(game.qtable[(hashable_key, game.current_player_idx)].values())
                    break

                num += 1

            if not found:
                #print("NEW STATE")
                possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
                from_pos, slide = possible_moves[np.random.randint(len(possible_moves))]
        else:
            #print("IN QTABLE")
            found = True
            hashable_key = hashable_key_cs
            max_Q = max(game.qtable[(hashable_key, game.current_player_idx)].values())
        
        if found:
            best_moves = []

            for i in game.qtable[(hashable_key, game.current_player_idx)]:
                if game.qtable[(hashable_key, game.current_player_idx)][i] == max_Q:
                    best_moves.append(i)

            from_pos, slide = best_moves[np.random.randint(len(best_moves))]

            if not cs_in_q_table:
                current_board = game.get_board()

                game._board = game.set_to_matrix(eq_state)
                game._Game__move(from_pos, slide, game.current_player_idx)

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

                next_state = game.matrix_to_set(next_board)

                moves = game.how_to_get_there(current_state, next_state)

                from_pos, slide = moves[np.random.randint(len(moves))]

                game._board = copy.deepcopy(current_board)

        return from_pos, slide
    
class RLGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.qtables = {}
        self.current_state = ([],[])
        self.next_state = ([],[])

    def matrix_to_set(self, board: np.ndarray):
        zero_set = []
        one_set  = []

        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if board[row][column] == 0:
                    zero_set.append((column, row))
                elif board[row][column] == 1:
                    one_set.append((column, row))

        return (sorted(zero_set), sorted(one_set))
    
    def set_to_matrix(self, state):
        board = np.ones((5, 5), dtype=np.uint8) * -1

        for s in state[0]:
            board[s[1]][s[0]] = 0

        for s in state[1]:
            board[s[1]][s[0]] = 1

        return board
    
    def initialize_qtable(self, possible_moves):
        moves_values = {}

        for j in possible_moves:
            moves_values[j] = 0.0

        return moves_values
    
    def rotate(self, board):
        rotated_states = []

        board1 = np.rot90(board)    #ROTATE 90 DEGREE
        rotated_states.append(self.matrix_to_set(board1))
        board2 = np.rot90(board, k=2) #ROTATE 180 DEGREE
        rotated_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(board, k=3)   #ROTATE 270 DEGREE
        rotated_states.append(self.matrix_to_set(board3))
                    
        return rotated_states

    def reflect(self, board):
        reflected_states = []

        board1 = np.fliplr(board)   #REFLECT HORIZONTALLY
        reflected_states.append(self.matrix_to_set(board1))
        board2 = np.flipud(board)   #REFLECT VERTICALLY
        reflected_states.append(self.matrix_to_set(board2))
        board3 = np.rot90(np.fliplr(board))    #REFLECT DIAGONALLY
        reflected_states.append(self.matrix_to_set(board3))
        board4 = np.rot90(np.flipud(board))    #REFLECT ANTI-DIAGONALLY
        reflected_states.append(self.matrix_to_set(board4))
                    
        return reflected_states

    def get_possible_moves(self, state, player_id):
        possible_from_pos = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)]
        possible_actions = []
        possible_moves = []

        if player_id == 0:
            possible_from_pos = [p for p in possible_from_pos if p not in state[1]]
        elif player_id == 1:
            possible_from_pos = [p for p in possible_from_pos if p not in state[0]]

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
    
    def how_to_get_there(self, current_state, next_state):
        moves = []
        current_board = copy.deepcopy(self.get_board())
        board = self.set_to_matrix(current_state)

        possible_moves = self.get_possible_moves(current_state, self.current_player_idx)

        for p in possible_moves:
            self._board = copy.deepcopy(board)
            self._Game__move(p[0], p[1], self.current_player_idx)

            state = self.matrix_to_set(self.get_board())

            if state[0] == next_state[0] and state[1] == next_state[1]:
                moves.append((p[0], p[1]))

        self._board = copy.deepcopy(current_board)

        return moves

    def update_Q_table(self, qtable_index, alpha, gamma, current_state, from_pos, slide, reward, next_state) -> None:
        possible_moves_cs = self.get_possible_moves(copy.deepcopy(current_state), self.current_player_idx)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        if (hashable_key_cs, self.current_player_idx) not in self.qtables[qtable_index]:
            self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)] = self.initialize_qtable(possible_moves_cs)

        next_player = (self.current_player_idx+1)%2
        hashable_key_ns = tuple(tuple(inner_tuple) for inner_tuple in next_state)
        if (hashable_key_ns, next_player) in self.qtables[qtable_index]:
            max_next_Q = max(self.qtables[qtable_index][(hashable_key_ns, next_player)].values())
        else:
            max_next_Q = 0.0

        #TIME DIFFERENCE
        self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] = (1 - alpha) * self.qtables[qtable_index][(hashable_key_cs, self.current_player_idx)][(from_pos, slide)] + alpha * (reward + gamma * max_next_Q)
        #MONTE CARLO
        #self.qtable[current_state][m] = self.qtables[qtable_index][current_state][m] + alpha * reward
    
    def train_agent(self) -> None:
        print("TRAINING")
        for _ in tqdm.tqdm(range(100_000)):
            self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD
            self.current_player_idx = 1 #RESET PLAYER INDEX
            player1 = RandomPlayer()
            player2 = RandomPlayer()
            self.playRL(True, player1, player2)

    def playRL(self, training: bool, player1: "Player", player2: "Player") -> int:
        '''Play the game. Update qtable. Returns the winning player'''
        players = [player1, player2]
        winner = -1

        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            from_pos = -1
            slide = -1

            self.current_state = self.matrix_to_set(self.get_board())

            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
                
            self.next_state = self.matrix_to_set(self.get_board())

            winner = self.check_winner()

            if training:
                for r in self.qtables:
                    if winner == -1:
                        reward = 0
                    elif winner == r:
                        reward = 1
                    else:
                        reward = -1

                    self.update_Q_table(r, 0.1, 0.9, self.current_state, from_pos, slide, reward, self.next_state)
            
        return winner


if __name__ == '__main__':
    num_of_matches = 100    #Number of matches to play
    num_wins_0 = 0          #Counter for player1 wins
    num_wins_1 = 0          #Counter for player2 wins

    #player1 = MyPlayerRL()
    #player1 = MyPlayerRLMinMax()
    player1 = MyPlayerMinMax()
    player2 = RandomPlayer()
    
    players = [player1, player2]
    
    #g = RLGame()
    #g = RLMinMaxGame()
    g = MinMaxGame()

    #WRITE Q-TABLE TO FILE
    #with open('quixo/100k-05-09-qtable.pkl', 'wb') as f:
     #   pickle.dump(g.qtable, f)
    #f.close()

    #LOAD Q-TABLE FROM FILE
    #with open('quixo/100k-01-09-qtable.pkl', 'rb') as f:
     #   g.qtable = pickle.load(f)
    #f.close()

    for player_index in range(len(players)):
        if isinstance(players[player_index], MyPlayerRLMinMax) or isinstance(players[player_index], MyPlayerRL):
            g.qtables[player_index] = defaultdict(list)
    
    #g.train_agent()

    print(f"PLAYING {num_of_matches} MATCHES")
    for _ in tqdm.tqdm(range(num_of_matches)):
        g._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD
        g.current_player_idx = 1 #RESET PLAYER INDEX

        #winner = g.playRL(False, player1, player2)
        #winner = g.playRLMinMax(player1, player2, testing=False)
        winner = g.playMinMax(player1, player2, testing=False)
        
        if winner == 0:
            num_wins_0 += 1
        elif winner == 1:
            num_wins_1 += 1
    
    print(f"WINS PLAYER 0: {num_wins_0}, WINS PLAYER 1: {num_wins_1}")
