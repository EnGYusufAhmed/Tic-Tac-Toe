from typing import Dict, Tuple
import random
import numpy as np
import joblib

State = Tuple[int,...]

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.Q: Dict[Tuple[State,int], float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state: State, action: int) -> float:
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state: State, available_actions, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        qs = [self.get_q(state,a) for a in available_actions]
        max_q = max(qs)
        max_actions = [a for a,q in zip(available_actions, qs) if q == max_q]
        return random.choice(max_actions)

    def update(self, state: State, action: int, reward: float, next_state: State, next_available_actions, done: bool):
        cur_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_qs = [self.get_q(next_state,a) for a in next_available_actions]
            target = reward + self.gamma * (max(next_qs) if next_qs else 0.0)
        new_q = cur_q + self.alpha * (target - cur_q)
        self.Q[(state, action)] = new_q

    def save(self, path: str):
        joblib.dump(self.Q, path)

    def load(self, path: str):
        self.Q = joblib.load(path)