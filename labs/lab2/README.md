# LAB 2 - NIM

The Nim Game is about ......

## GOALS

1) Find an Evolution Strategy for the problem

Before starting to implement the ES, I tried doing some matches to benchmark the goodness of the template given strategies.
The benchmarks are based on a total of 100 matches.
The optimal strategy gets to achieve around 70% of winning probability, which is good...

![immagine](https://github.com/andrea-ga/computational-intelligence/assets/55812399/464fea43-4c7d-4f02-80ae-d55471033db9)

..But I couldn't stop wondering if there was any way to improve it.
Searching the internet about informations, I found a way to implement a 100% winning strategy (Thanks to Wikipedia).

The strategy is based on the definition of a special state.
</br>
Special state: there's only one row that has a number of elements greater or equal to 2.
The winning strategy is achieved by following two rules:
  1. In normal play, we always try to make a move which result in a nim sum equals to 0.
  2. When we get to the special state, our goal is to reach a state with an odd number of 1 element rows.
The second rule is essential, and will make the opponent lose.

![immagine](https://github.com/andrea-ga/computational-intelligence/assets/55812399/c3f26972-177f-4d7c-a0d4-ff9189aa54fd)


### ES
