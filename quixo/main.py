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

#MINMAX
class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MinMaxGame', alpha, beta, f, visited_states, depth, max_depth) -> tuple[int, tuple[tuple[int, int], Move]]:
        winner = game.check_winner()
        opt_move = ()
        current_state = copy.deepcopy(game.current_state)
        current_board = copy.deepcopy(game.get_board())
        counter = 0

        if depth >= max_depth: #HARD CUT OFF
            f.write(f"###################CUT OFF####################\n")
            return (0, opt_move)

        if winner != -1:
            if winner == 0:
                reward = 1
            else:
                reward = -1
            f.write(f"**********WINNER REACHED - PLAYER {winner}*************\n")
            return (reward, opt_move)
        
        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)
        possible_moves = game.remove_symmetries(possible_moves, game.current_player_idx, current_board)
        
        if game.current_player_idx == 0:
            max_value = -math.inf
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game.moveMinMax(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                f.write(f"STATE: {game.current_state}\n")
                f.write(f"BOARD:\n{game.get_board()}\n")

                if (game.current_state, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    game.current_state = copy.deepcopy(current_state)
                    game._board = copy.deepcopy(current_board)
                    f.write("_________ALREADY VISITED STATE___________\n")
                    continue

                visited_states.append((game.current_state, game.current_player_idx))
                
                game.current_player_idx = 1
                reward, _ = self.make_move(game, alpha, beta, f, visited_states, depth+1, max_depth)
                counter += reward
                game.current_player_idx = 0
                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if counter > max_value:
                    max_value = counter
                    opt_move = child_state
                
                alpha = max(alpha, max_value)
                if alpha >= beta:
                    f.write("---------------PRUNING-----------------\n")
                    break

            return (max_value, opt_move)
        elif game.current_player_idx == 1:
            min_value = math.inf
            while len(possible_moves) != 0:
                child_state = possible_moves[random.randint(0, len(possible_moves)-1)]
                possible_moves.remove(child_state)
                from_pos, slide = child_state
                game.moveMinMax(from_pos, slide, game.current_player_idx)
                game.current_state = game.matrix_to_set(game.get_board())

                f.write(f"PLAYER {game.current_player_idx} DID <{from_pos}, {slide}>\n")
                f.write(f"STATE: {game.current_state}\n")
                f.write(f"BOARD:\n{game.get_board()}\n")

                if (game.current_state, game.current_player_idx) in visited_states:   #DON'T RETURN TO A PREVIOUS STATE
                    game.current_state = copy.deepcopy(current_state)
                    game._board = copy.deepcopy(current_board)
                    f.write("_________ALREADY VISITED STATE___________\n")
                    continue

                visited_states.append((game.current_state, game.current_player_idx))

                game.current_player_idx = 0
                reward, _ = self.make_move(game, alpha, beta, f, visited_states, depth+1, max_depth)
                counter += reward
                game.current_player_idx = 1
                game.current_state = copy.deepcopy(current_state)
                game._board = copy.deepcopy(current_board)

                if counter < min_value:
                    min_value = counter
                    opt_move = child_state

                beta = min(beta, min_value)
                if beta <= alpha:
                    f.write("---------------PRUNING-----------------\n")
                    break

            return (min_value, opt_move)


class MinMaxGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.current_state = ([], [])

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
            self.moveMinMax(s[0], s[1], player_id)

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

    def playMinMax(self, testing: bool, player1: "Player", player2: "Player") -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        num_moves = 0
        self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD

        #if testing:
            #print("START")
            #self.print()

        while winner < 0:
            num_moves += 1
            if num_moves > 1_000:
                break

            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            from_pos = -1
            slide = -1
            self.current_state = self.matrix_to_set(self.get_board())

            if testing:
                f = open("quixo/play.txt", "w")
                f.write("^^^^^^^^^^^START^^^^^^^^^^^\n")
                f.write(f"STATE: {self.current_state}\n")
                f.write(f"BOARD:\n{self.get_board()}\n")
                f.close()
                f = open("quixo/play.txt", "a")
            
            visited_states = []

            while not ok:
                if self.current_player_idx == 0:
                    _, opt_move = players[self.current_player_idx].make_move(self, -math.inf, math.inf, f, visited_states, 0, 10)
                else:
                    opt_move = players[self.current_player_idx].make_move(self)

                if testing:
                    f.write(f"OPT_MOVE: {opt_move}")
                    f.close()
                    exit(1)

                ok = self._Game__move(opt_move[0], opt_move[1], self.current_player_idx)

            #if testing:
                #print(f"Player: {self.current_player_idx}")
                #self.print()
            
        return winner
        



#REINFORCEMENT LEARNING (QLEARNING)
class MyPlayer2(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'RLGame') -> tuple[tuple[int, int], Move]:
        current_state = copy.deepcopy(game.current_state)
        possible_moves = game.get_possible_moves(current_state, game.current_player_idx)

        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        if (hashable_key_cs, game.current_player_idx) not in game.qtable:
            game.qtable[(hashable_key_cs, game.current_player_idx)] = game.initialize_qtable(possible_moves)

        max_Q = max(game.qtable[(hashable_key_cs, game.current_player_idx)].values())
        
        best_moves = []

        for i in game.qtable[(hashable_key_cs, game.current_player_idx)]:
            if game.qtable[(hashable_key_cs, game.current_player_idx)][i] == max_Q:
                best_moves.append(i)

        from_pos, slide = best_moves[np.random.randint(len(best_moves))]

        return from_pos, slide
    
class RLGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.qtable = defaultdict(list)
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

    def update_Q_table(self, alpha, gamma, current_state, moves, reward, next_state) -> None:
        possible_moves_cs = self.get_possible_moves(copy.deepcopy(current_state), self.current_player_idx)
        hashable_key_cs = tuple(tuple(inner_tuple) for inner_tuple in current_state)
        if (hashable_key_cs, self.current_player_idx) not in self.qtable:
            self.qtable[(hashable_key_cs, self.current_player_idx)] = self.initialize_qtable(possible_moves_cs)

        next_player = (self.current_player_idx+1)%2
        possible_moves_ns = self.get_possible_moves(copy.deepcopy(next_state), next_player)
        hashable_key_ns = tuple(tuple(inner_tuple) for inner_tuple in next_state)
        if (hashable_key_ns, next_player) not in self.qtable:
            self.qtable[(hashable_key_ns, next_player)] = self.initialize_qtable(possible_moves_ns)

        max_next_Q = max(self.qtable[(hashable_key_ns, next_player)].values())

        for m in moves:
            #TIME DIFFERENCE
            self.qtable[(hashable_key_cs, self.current_player_idx)][m] = (1 - alpha) * self.qtable[(hashable_key_cs, self.current_player_idx)][m] + alpha * (reward + gamma * max_next_Q)
            #MONTE CARLO
            #self.qtable[current_state][m] = self.qtable[current_state][m] + alpha * reward
    
    def playRL(self, player1: "Player", player2: "Player") -> int:
        '''Play the game. Update qtable. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        num_moves = 0
        self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD

        #print("START")
        #self.print()

        while winner < 0:
            num_moves += 1
            if num_moves > 1_000:
                break

            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            from_pos = -1
            slide = -1

            current_states = []
            next_states = []
            moves_list = []

            self.current_state = self.matrix_to_set(self.get_board())
            current_states.append(self.current_state)
            for cs in self.rotate(self.get_board()):
                current_states.append(cs)
            for cs in self.reflect(self.get_board()):
                current_states.append(cs)

            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)

            #print(self.current_player_idx, from_pos, slide)

            self.next_state = self.matrix_to_set(self.get_board())
            next_states.append(self.next_state)
            for ns in self.rotate(self.get_board()):
                next_states.append(ns)
            for ns in self.reflect(self.get_board()):
                next_states.append(ns)

            winner = self.check_winner()
            reward = 0
            if winner == 0:
                reward = 1
            elif winner == 1:
                reward = -1

            for c in zip(current_states, next_states):
                moves = self.how_to_get_there(c[0], c[1])
                moves_list.append(moves)
            
            for c in zip(current_states, next_states, moves_list):
                self.update_Q_table(0.1, 0.9, c[0], c[2], reward, c[1])

            #print(f"Player: {self.current_player_idx}")
            #self.print()
            
        return winner

    def train_agent(self) -> None:
        print("TRAINING")
        for _ in tqdm.tqdm(range(10_000)):
            player1 = RandomPlayer()
            player2 = RandomPlayer()
            self.playRL(player1, player2)
            #self.print()


if __name__ == '__main__':
    g = RLGame()
    #g.train_agent()
    #g = MinMaxGame()

    #with open('quixo/10k-01-09-qtable.pkl', 'wb') as f:
     #   pickle.dump(g.qtable, f)

    #LOAD Q-TABLE FROM FILE
    with open('quixo/10k-01-09-qtable.pkl', 'rb') as f:
        g.qtable = pickle.load(f)
    f.close()

    #print(g.qtable)

    player1 = MyPlayer2()
    player2 = RandomPlayer()

    num_wins_0 = 0
    num_wins_1 = 0
    num_ties = 0

    print(f"PLAYING")
    for _ in tqdm.tqdm(range(100)):
        winner = g.play(player1, player2)
        if winner == 0:
            num_wins_0 += 1
        elif winner == 1:
            num_wins_1 += 1
        elif winner == -1:
            num_ties += 1
    
    print(f"WINS PLAYER 0: {num_wins_0}, WINS PLAYER 1: {num_wins_1}, TIES: {num_ties}")
