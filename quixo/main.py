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

    def make_move(self, game: 'MinMaxGame', state, player_id) -> tuple[int, tuple[tuple[int, int], Move]]:
        winner = game.check_winner()
        opt_state = ()

        if winner != -1:
            print("WINNER REACHED")
            return (winner, opt_state)
        
        children_states = game.get_children_states(state, player_id)
        print(children_states)

        if player_id == 0:
            max_value = -math.inf
            for child_state in children_states.keys():
                #if state.is_equivalent(child_state):
                  #  continue  # Don't explore equivalent states
                
                reward, _ = self.make_move(game, child_state, player_id)

                print(reward)
                if max_value < reward:
                    max_value = reward
                    opt_state = children_states[child_state]
                    print("HERE1")
                    print(opt_state)

                return (max_value, opt_state)
        elif player_id == 1:
            min_value = math.inf
            for child_state in children_states.keys():
                #if state.is_equivalent(child_state):
                 #   continue  # Don't explore equivalent states

                reward, _ = self.make_move(game, child_state, player_id)

                if min_value > reward:
                    min_value = reward
                    opt_state = children_states[child_state]
                    print("HERE2")
                    print(opt_state)

                return (min_value, opt_state)

class MinMaxGame(Game):
    def __init__(self) -> None:
        super().__init__()

    def matrix_to_set(self, board: np.ndarray):
        zero_set = []
        one_set  = []

        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if board[row][column] == 0:
                    zero_set.append((column, row))
                elif board[row][column] == 1:
                    one_set.append((column, row))

        return (frozenset(zero_set), frozenset(one_set))

    def get_children_states(self, state, player_id):
        possible_from_pos = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(4,1),(0,2),(4,2),(0,3),(4,3),(0,4),(1,4),(2,4),(3,4),(4,4)]
        possible_actions = []
        children_states = {}

        tmp_board = self.get_board()

        if player_id == 0:
            possible_from_pos = [p for p in possible_from_pos if p not in state[1]]

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
                    self.moveMinMax(p, s, player_id)

                    children_states[self.matrix_to_set(self.get_board())] = (p, s)

                    self._board = tmp_board
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
                    self.moveMinMax(p, s, player_id)

                    children_states[self.matrix_to_set(self.get_board())] = (p, s)

                    self._board = tmp_board

        return children_states

    def moveMinMax(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        if player_id > 1:
            print("WRONG PLAYER INDEX")
            return False
        
        prev_value = copy.deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.takeMinMax((from_pos[1], from_pos[0]), player_id)

        if acceptable:
            acceptable = self._Game__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = copy.deepcopy(prev_value)
                
        return acceptable
    
    def takeMinMax(self, from_pos: tuple[int, int], player_id: int) -> bool:
        acceptable: bool = (
            (from_pos[0] < 5 and from_pos[1] == 0)
            or (from_pos[0] < 5 and from_pos[1] == 4)
            or (from_pos[1] < 5 and from_pos[0] == 0)
            or (from_pos[1] < 5 and from_pos[0] == 4)
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id

        return acceptable

    def playMinMax(self, training: bool, player1: "Player", player2: "Player") -> int:
        '''Play the game. Update qtable. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        num_moves = 0
        self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD

        #if not training:
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

            while not ok:
                _, opt_state = players[self.current_player_idx].make_move(self, self.current_state, self.current_player_idx)
                print(opt_state)
                from_pos, slide = opt_state
                ok = self.moveMinMax(from_pos, slide, self.current_player_idx)

            self.next_state = self.matrix_to_set(self.get_board())

            #if not training:
                #print(f"Player: {self.current_player_idx}")
                #self.print()
            
        return winner
        



#REINFORCEMENT LEARNING (QLEARNING)
class MyPlayer2(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'RLGame') -> tuple[tuple[int, int], Move]:
        if game.current_state not in game.qtable:
            game.qtable[game.current_state] = np.zeros(64)

        not_acceptable = [0, 2, 4, 8, 12, 16, 19, 22, 27, 30, 35, 38, 43, 45, 46, 49, 53, 57, 61, 63]

        max_Q = np.max(np.array([x for i, x in enumerate(game.qtable[game.current_state]) 
                                 if i not in not_acceptable and i not in game.current_state[1]]))
        
        best_actions = []

        for i, q_value in enumerate(game.qtable[game.current_state]):
            if q_value == max_Q and i not in not_acceptable and i not in game.current_state[1]:
                best_actions.append(i)

        action = best_actions[np.random.randint(len(best_actions))]

        moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]

        if action < 4:
            from_pos = (0, 0)
            move = moves[action]
        elif action < 8:
            from_pos = (1, 0)
            move = moves[action - 4]
        elif action < 12:
            from_pos = (2, 0)
            move = moves[action - 8]
        elif action < 16:
            from_pos = (3, 0)
            move = moves[action - 12]
        elif action < 20:
            from_pos = (4, 0)
            move = moves[action - 16]
        elif action < 24:
            from_pos = (0, 1)
            move = moves[action - 20]
        elif action < 28:
            from_pos = (4, 1)
            move = moves[action - 24]
        elif action < 32:
            from_pos = (0, 2)
            move = moves[action - 28]
        elif action < 36:
            from_pos = (4, 2)
            move = moves[action - 32]
        elif action < 40:
            from_pos = (0, 3)
            move = moves[action - 36]
        elif action < 44:
            from_pos = (4, 3)
            move = moves[action - 40]
        elif action < 48:
            from_pos = (0, 4)
            move = moves[action - 44]
        elif action < 52:
            from_pos = (1, 4)
            move = moves[action - 48]
        elif action < 56:
            from_pos = (2, 4)
            move = moves[action - 52]
        elif action < 60:
            from_pos = (3, 4)
            move = moves[action - 56]
        elif action < 64:
            from_pos = (4, 4)
            move = moves[action - 60]

        return from_pos, move
    
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

        return (frozenset(zero_set), frozenset(one_set))

    def update_Q_table(self, alpha, gamma, current_state, from_pos, slide, reward, next_state) -> None:
        if current_state not in self.qtable:
            self.qtable[current_state] = np.zeros(64)
        if next_state not in self.qtable:
            self.qtable[next_state] = np.zeros(64)

        if from_pos == (0, 0):
            action = 0
        elif from_pos == (1, 0):
            action = 4
        elif from_pos == (2, 0):
            action = 8
        elif from_pos == (3, 0):
            action = 12
        elif from_pos == (4, 0):
            action = 16
        elif from_pos == (0, 1):
            action = 20
        elif from_pos == (4, 1):
            action = 24
        elif from_pos == (0, 2):
            action = 28
        elif from_pos == (4, 2):
            action = 32
        elif from_pos == (0, 3):
            action = 36
        elif from_pos == (4, 3):
            action = 40
        elif from_pos == (0, 4):
            action = 44
        elif from_pos == (1, 4):
            action = 48
        elif from_pos == (2, 4):
            action = 52
        elif from_pos == (3, 4):
            action = 56
        elif from_pos == (4, 4):
            action = 60        

        action += slide.value

        max_next_Q = np.max(self.qtable[next_state])
        #TIME DIFFERENCE
        self.qtable[current_state][action] = (1 - alpha) * self.qtable[current_state][action] + alpha * (reward + gamma * max_next_Q)
        #MONTE CARLO
        #self.qtable[current_state][action] = self.qtable[current_state][action] + alpha * reward

    def moveRL(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        if player_id > 1:
            print("WRONG PLAYER INDEX")
            return False
        
        prev_value = copy.deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.takeRL((from_pos[1], from_pos[0]), player_id)

        if acceptable:
            acceptable = self.slideRL((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = copy.deepcopy(prev_value)
                
        return acceptable
    
    def takeRL(self, from_pos: tuple[int, int], player_id: int) -> bool:
        acceptable: bool = (
            (from_pos[0] < 5 and from_pos[1] == 0)
            or (from_pos[0] < 5 and from_pos[1] == 4)
            or (from_pos[1] < 5 and from_pos[0] == 0)
            or (from_pos[1] < 5 and from_pos[0] == 4)
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id

        return acceptable
    
    def slideRL(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
    
    def playRL(self, training: bool, player1: "Player", player2: "Player") -> int:
        '''Play the game. Update qtable. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        num_moves = 0
        self._board = np.ones((5, 5), dtype=np.uint8) * -1 #RESET BOARD

        if not training:
            print("START")
            self.print()

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

            #if not training:
                #if self.current_state not in self.qtable:
                 #   print("NOT IN QTABLE")
                #else:
                  #  print(self.qtable[self.current_state])

            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(self)
                ok = self.moveRL(from_pos, slide, self.current_player_idx)

            if not training:
                print(self.current_player_idx, from_pos, slide)

            winner = self.check_winner()
            reward = 0
            if winner == 0:
                reward = 1
            elif winner == 1:
                reward = -1

            self.next_state = self.matrix_to_set(self.get_board())

            self.update_Q_table(0.2, 1, self.current_state, from_pos, slide, reward, self.next_state)

            if not training:
                print(f"Player: {self.current_player_idx}")
                self.print()
            
        return winner

    def train_agent(self) -> None:
        for _ in tqdm.tqdm(range(10_000)):
            player1 = RandomPlayer()
            player2 = RandomPlayer()
            self.playRL(True, player1, player2)
            #self.print()


if __name__ == '__main__':
    #g = RLGame()
    #g.train_agent()
    g = MinMaxGame()

    #print(g.qtable)

    #with open('quixo/05-05-qtable.pkl', 'wb') as f:
     #   pickle.dump(g.qtable, f)

    #LOAD Q-TABLE FROM FILE
    #with open('quixo/mc-05-qtable.pkl', 'rb') as f:
     #   g.qtable = pickle.load(f)

    player1 = MyPlayer()
    player2 = RandomPlayer()

    num_wins_0 = 0
    num_wins_1 = 0
    num_ties = 0

    for _ in tqdm.tqdm(range(100)):
        winner = g.playMinMax(False, player1, player2)
        if winner == 0:
            num_wins_0 += 1
        elif winner == 1:
            num_wins_1 += 1
        elif winner == -1:
            num_ties += 1
    
    print(f"WINS PLAYER 0: {num_wins_0}, WINS PLAYER 1: {num_wins_1}, TIES: {num_ties}")
