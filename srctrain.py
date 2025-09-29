import random
from src.env import TicTacToe
from src.agent import QLearningAgent


def random_policy(env, player):
    return random.choice(env.available_actions())


def train(episodes=20000, alpha=0.5, gamma=0.9, epsilon=0.1):
    env = TicTacToe()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    for ep in range(1, episodes+1):
        state, _ = env.reset()
        done = False
        current_player = 1  # agent starts
        while not done:
            available = env.available_actions()
            if current_player == 1:
                action = agent.choose_action(state, available, training=True)
                next_state, reward, done, info = env.step(action, 1)
                if done:
                    agent.update(state, action, reward, next_state, [], done)
                    break
                # opponent move
                opp_action = random.choice(env.available_actions())
                next_state2, reward2, done2, info2 = env.step(opp_action, -1)
                if done2:
                    # from agent perspective, losing if winner == -1
                    final_reward = -1.0 if info2.get('winner') == -1 else 0.5
                    agent.update(state, action, final_reward, tuple(env.board), [], True)
                    break
                else:
                    # intermediate update with zero reward
                    agent.update(state, action, 0.0, tuple(env.board), env.available_actions(), False)
                    state = tuple(env.board)
            else:
                # we always start with agent as player 1 in this training loop
                pass

        if ep % 1000 == 0:
            print(f"Episode {ep}/{episodes}")

    # save model
    import os
    os.makedirs('models', exist_ok=True)
    agent.save('models/qtable.pkl')
    print('Training finished. Q-table saved to models/qtable.pkl')

if __name__ == '__main__':
    train(episodes=20000)