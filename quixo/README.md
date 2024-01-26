Author: Andrea Galella s310166
</br>
E-mail: <andrea.galella@studenti.polito.it> or <galella.andrea@gmail.com>
</br>
Co-author: Prof. Andrea Calabrese
</br>
### References:
  - Class lectures and slides
  - [Course repository](https://github.com/squillero/computational-intelligence)
  - [geeks for geeks for alpha-beta pruning algorithm](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/)

#  MinMax for Quixo

To reduce the number of states to visit at each run, the algorithm:
- Removes all the board symmetries;
- Keeps track of the already visited states, in order to have a quick reward result from them;
- Uses alpha-beta pruning, to prune not promising branches;
- Uses hard cut off, in order to be faster.

Increasing the *max_depth* parameter (inside the *make_move* function of the *MyPlayerMinMax* class) allows to have better results but with longer computational time. This is because the hard cut off is called later, allowing the agent to discover more states.

## Results
### MinMax (player 0) vs Random (player 1) with *max_depth=1*
![MinMax Agent play first vs Random with max_depth=1](https://github.com/andrea-ga/computational-intelligence/blob/main/quixo/img/100-vsRandom-1.png)

### Random (player 0) vs MinMax (player 1) with *max_depth=1*
![MinMax Agent play second vs Random with max_depth=1](https://github.com/andrea-ga/computational-intelligence/blob/main/quixo/img/100-Randomvs-1.png)

### MinMax (player 0) vs Random (player 1) with *max_depth=2*
![MinMax Agent play first vs Random with max_depth=2](https://github.com/andrea-ga/computational-intelligence/blob/main/quixo/img/100-vsRandom-2.png)

### Random (player 0) vs MinMax (player 1) with *max_depth=2*
![MinMax Agent play second vs Random with max_depth=2](https://github.com/andrea-ga/computational-intelligence/blob/main/quixo/img/100-Randomvs-2.png)

# Other Attempts
## Reinforcement Learning (Q-learning)