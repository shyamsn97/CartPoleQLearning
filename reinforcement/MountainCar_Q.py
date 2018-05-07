import gym
import numpy as np


class Mountain_Climber():

	def __init__(self):
		self.env = gym.make('MountainCar-v0')
		self.epsilon = 1
		self.decay = 0.99
		self.gamma = 0.8
		self.learning_rate = 0.05
		self.weights =  np.random.random(size=(2,3))
		self.trials = 500
		self.trial_len = 500

	def predict(self,state_vec):

		if self.epsilon >= 0.01:
			self.epsilon *= self.decay
			if np.random.random() < self.epsilon:
				action = self.env.action_space.sample()
				print("RANDOM MOVE with EPSILON: " + str(self.epsilon))
				return action, state_vec.dot(self.weights)[action]
		return np.argmax(state_vec.dot(self.weights)), np.max(state_vec.dot(self.weights))

	def update(self,difference,state_vec,action):
		self.weights[:,action] += self.learning_rate*difference*state_vec

	def main(self):

		for trial in range(self.trials):
			curr = self.env.reset()
			print("TRIAL: " + str(trial))
			for steps in range(self.trial_len):
				self.env.render()
				action, qval = self.predict(curr)

				next_state, reward, done, _ = self.env.step(action)
				# reward = reward + 100*next_state[1] - steps
				newaction, newq = self.predict(next_state)
				if done:
					diff = reward - qval
				else:
					diff = reward + self.gamma*newq - qval

				self.update(diff,next_state,action)
				curr = next_state
				if done:
					print("Finished at step: " + str(steps))
					print(self.weights)
					break

env = gym.make('MountainCar-v0')
# curr = env.reset()
# print(curr)
# for i in range(500):
# 	env.render()
# 	state, reward, done, _ = env.step(0)
mc = Mountain_Climber()
mc.main()
