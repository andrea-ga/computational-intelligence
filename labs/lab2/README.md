Author: Andrea Galella s310166

### References:
  - [Wikipedia](https://en.wikipedia.org/wiki/Nim)
  - [Course repository](https://github.com/squillero/computational-intelligence)

# LAB 2 - NIM

*Nim is a mathematical game of strategy in which two players take turns removing (or "nimming") objects from distinct heaps or piles. On each turn, a player must remove at least one object, and may remove any number of objects provided they all come from the same heap or pile. Depending on the version being played, the goal of the game is either to avoid taking the last object or to take the last object.* (Wikipedia)

The version of the game we're going to play with is the one where the goal is to avoid taking the last object, also called the *misére game* version.

Definitions:
  1. `state`: An instance of the game. Each instance is defined by the total number of objects remaining in each row. 
  2. `nim sum`: sum of the bitwise XOR of the number of elements of each row.

## GOALS

1) Find an Evolution Strategy for the problem

Before starting to implement the ES, I tried doing some matches to benchmark the goodness of the template given strategies.
</br>
`All benchmarks are created from Nim(5) and played againts the pure_random agent on a total number of 100 matches.`
</br>
The optimal strategy gets to achieve around 70% of winning probability, which is good...

![immagine](https://github.com/andrea-ga/computational-intelligence/assets/55812399/464fea43-4c7d-4f02-80ae-d55471033db9)

..But I started wondering if there was any way to improve it.
Searching the internet about informations, I found a way to implement a 100% winning strategy (Thanks to Wikipedia).

The strategy is based on the definition of a special state.
</br>
Special state: there's only one row that has a number of elements greater or equal to 2.
</br>
The winning strategy is achieved by following two rules:
  1. In normal play, we always try to make a move that results in a state with a nim sum equals to 0.
  2. When we get to the special state, our goal is to reach a state with an odd number of 1 element rows.
</br>
The second rule is essential, and will make the opponent lose.

![immagine](https://github.com/andrea-ga/computational-intelligence/assets/55812399/c3f26972-177f-4d7c-a0d4-ff9189aa54fd)


### ES
For the implementation of the Evolution Strategy I chose to implement a (1 + λ) strategy. Where lambda is the number of children created at each generation (iteration). In this type of strategy the resulting population at the end of each generation is given by: λ children + parent. From here we select the individual with the best fitness to be the parent of the next generation.

An Individual inside a population is composed by the following attributes:
  1. `row`: an integer that indicates from which row we want to remove the objects.
  2. `num_obj`: a list of integer. This list contains all the possible number of objects we can remove from that row.
  3. `weights`: an array of float. Each weight is associated, by the index, to a num_obj element.
  4. `strategy`: a Nimply object that puts together the row and number of objects to take.
  5. `fitness`: an integer number that indicates the evaluation of the move. Greater is the value, better is the move.

For each state, we start with populating the population by selecting one Individual for each row.
</br>
Then we select for each individual the number of objects to take from that row. This is done by selecting one of the value in the num_obj list, based on a probability given by the weights array -> bigger the weight value, higher the probability to get selected. 

The weights are random values, that gets mutated with a gaussian with fixed values mu = 0 and sigma = 0.01. The mutation rate is fixed at the value of 0.5.

A strategy is created by putting together the row with the selected number of objects.
</br>
From this we can evaluate the fitness value of the move. A fitness can have 5 different values:
  1) `fitness = 0`  -> Not a very interesting move.
  2) `fitness = 1`  -> A move that achieves a nim sum of 0. Best move for the great part of the game.
  3) `fitness = 2`  -> We're in the special state. The row is the right one, but the number of objects is not correct.
  4) `fitness = 3`  -> We're in the special state. Right row and right number of objects.
  5) `fitness = -1` -> We're in a row with 0 elements.

This ES (with λ = 5, 1_000/λ generations, ) gets to achieve around the 95% of winning probability:

![immagine](https://github.com/andrea-ga/computational-intelligence/assets/55812399/ba3013ba-3289-4c22-b6eb-61709a00fa0c)



