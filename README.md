# CartPoleQLearning

## Q learning agent that learns to master Cart Pole system

![alt text](game.png)

### game stops after 500 steps

### This agent consistently solves the system (gets an average of 195 or greater reward over 100 consecutive trials) at around trial 120-150

## Linear function and optimization process

![alt text](formula.png)

## Clip of agent learning to play

![alt text](Learning.gif)

## Clip of perfect solution found by agent (Achieves maximum 500 reward every trial)

![alt text](PerfectSolution.gif)

## How to use 
  (requires python3) <br>
  Show help: python LinearQCartPole.py -h <br>
  Train a new agent: python LinearQCartPole.py [episodes]  <br>
  Random Agent: python LinearQCartPole.py [episodes] -random  <br>
  Load in weights: python LinearQCartPole.py [episodes] -f [filename.txt]   <br>
  Test to see average solve times: python LinearQCartPole.py [episodes] -test [number of tests] 

	
