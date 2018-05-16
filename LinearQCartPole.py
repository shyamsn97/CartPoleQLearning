import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys

class LinearQCartPole():

	def __init__(self,trials):

		self.env = gym.make('CartPole-v1')
		self.epsilon = 1
		self.decay = 0.995
		self.gamma = 1
		self.learning_rate = 0.05
		self.weights =  np.zeros(8)
		self.trials = trials
		self.trial_len = 500
		self.converge = False
		self.solved = False

	def initialize(self):

		self.epsilon = 1
		self.decay = 0.995
		self.gamma = 1
		self.learning_rate = 0.05
		self.weights =  np.zeros(8)
		self.converge = False
		self.solved = False

	def get_q_vals(self,state_vec):

		state_vec1 = self.transform(state_vec,0)
		state_vec2 = self.transform(state_vec,1)
		q = []
		q.append(state_vec1.dot(self.weights))
		q.append(state_vec2.dot(self.weights))
		return np.array(q)

	def predict(self,state_vec,action=False):

		qvals = self.get_q_vals(state_vec)
		if self.epsilon >= 0.01:
			self.epsilon *= self.decay
			if np.random.random() < self.epsilon:
				action = self.env.action_space.sample()
				print("RANDOM MOVE with EPSILON: " + str(self.epsilon))
				return action, qvals[action]
		return np.argmax(qvals), np.max(qvals)

	def update(self,difference,state_vec,action):

		state_vec = self.transform(state_vec,action)
		self.weights -= self.learning_rate*difference*state_vec

	def transform(self,vec,action):

		zeros = np.zeros(8)
		if action == 0:
			zeros[0:4] = vec
		else:
			zeros[4:] = vec
		return zeros

	def main(self,random=False,file="none",test=False,plot=True,render=True):

		self.initialize()

		if file != "none":
			self.weights = np.loadtxt("weights/" + file)
			self.epsilon = 0
		scores = deque(maxlen=100)
		meanscores = deque(maxlen=self.trials)
		maxstep = 0
		rew_array = []
		cumulative_rew = 0
		converge_point = 0

		for trial in range(self.trials):
			curr = self.env.reset()
			cumulative = 0
			for steps in range(self.trial_len):

				if render ==  True:
					self.env.render()
				if random == True:
					action = self.env.action_space.sample()
					next_state, reward, done, _ = self.env.step(action)
					curr = next_state
				elif file != "none":
					action, qval = self.predict(curr)
					next_state, reward, done, _ = self.env.step(action)
					curr = next_state
				else:
					action, qval = self.predict(curr)
					next_state, reward, done, _ = self.env.step(action)
					newaction, newq = self.predict(next_state)
					if done:
						diff = reward - qval
					else:
						diff = reward + self.gamma*newq - qval
						if steps == 498:
							if trial >= 150:
								# self.converge = True
								self.learning_rate = 0.01
						if self.converge != True:
							self.update(diff,next_state,action)
					curr = next_state

				if done:
					print("---------------------------------")
					if steps > maxstep:
						maxstep = steps
					print("Finished at step: " + str(steps))
					print("Max Step Reached: " + str(maxstep))
					print("Weights: " + str(self.weights))
					cumulative += reward
					print("Reward : " + str(cumulative))
					break
				cumulative += reward

			scores.append(cumulative)
			print("Mean Reward: " + str(np.mean(scores)))
			meanscores.append(np.mean(scores))

			print("TRIAL: " + str(trial))

			if np.mean(scores) >= 195 and trial >= 100:
				self.solved = True

			if self.solved == True:
				if converge_point == 0:
					converge_point = trial
				print("SOLVED at trial: " + str(converge_point))
				if test == True:
					return converge_point

			cumulative_rew += cumulative
			rew_array.append(cumulative)

			if self.converge == True:
				print("CONVERGED")

		print("AVG REWARD: " + str(cumulative_rew/self.trials))

		if random == False and file == "none":
			print("Save File")
			np.savetxt('weights/weights.txt',self.weights)

		if plot == True:
			x = np.arange(self.trials)
			m, b = np.polyfit(x, np.array(rew_array), 1)
			plt.plot(rew_array,label='Reward',color='indigo')
			plt.plot(x, m*x + b, '-',label='Regression Line',color='orange')
			plt.plot(np.array(meanscores), '-', label='Rolling Mean (window = 100)',color='darkgreen')
			if converge_point != 0:
				plt.axvline(x=converge_point, linestyle='dashed', label='Solved',color='red')
			legend = plt.legend(loc='upper left', shadow=True)
			frame = legend.get_frame()
			plt.ylabel('Reward')
			plt.xlabel('Episodes')
			plt.ylim(0, np.max(rew_array)+20)
			plt.show()

		return converge_point

	def test(self,ran):
		solved_times = []
		for i in range(int(ran)):
			converge = self.main(test=True,plot=False,render=False)
			solved_times.append(converge)
			print("FINISHED TRIAL: " + str(i))
		print("AVERAGE SOLVE TIME: " + str(np.mean(solved_times)))

if __name__ == '__main__':

	quiet = False
	if "-quiet" in sys.argv:
		quiet = True
	if len(sys.argv) > 2:
		if sys.argv[2] == "-random":
			mc = LinearQCartPole(trials=int(sys.argv[1]))
			mc.main(random=True)

		elif sys.argv[2] == "-f":
			mc = LinearQCartPole(trials=int(sys.argv[1]))
			mc.main(file=sys.argv[3])

		elif sys.argv[2] == "-test":
			mc = LinearQCartPole(trials=int(sys.argv[1]))
			mc.test(sys.argv[3])

	elif len(sys.argv) == 2:
		if len(sys.argv) == 2 and sys.argv[1] == "-h":
			s = "This agent solves the CartPole environment(average reward of >= 195 over 100 consecutive trials) using linear Q learning(Online Least Squares)\n   Arguments:\n   Train a new agent: [episodes]\n   Random Agent: [episodes] -random\n   Load in weights: [episodes] -f [filename.txt]\n   Test to see average solve times: [episodes] -test [number of tests]"
			print(s)
		elif sys.argv[1].isdigit():
			mc = LinearQCartPole(trials=int(sys.argv[1]))
			mc.main()
		else:
			print("Please enter a number of episodes!")

	else:
		print("Please enter a number of episodes or a command line argument!")
