from src.env import TicTacToe
from src.agent import QLearningAgent


def human_move(env: TicTacToe):
    env.render()
    moves = env.available_actions()
    print('Available:', moves)
    while True:
        try:
            a = int(input('Your move (0-8): '))
            if a in moves:
                return a
            else:
                print('Invalid move')
        except Exception:
            print('Enter a number 0-8')


def play():
    env = TicTacToe()
    agent = QLearningAgent()
    try:
        agent.load('models/qtable.pkl')
        print('Loaded Q-table from models/qtable.pkl')
    except Exception as e:
        print('Could not load Q-table. Play will use a random agent. Error:', e)

    state, _ = env.reset()
    while True:
        # Human is O (-1). Let agent be X (1) and start first.
        action = agent.choose_action(state, env.available_actions(), training=False)
        state, reward, done, info = env.step(action, 1)
        print('\nAgent moves:')
        env.render()
        if done:
            print('Result:', info)
            break

        # human turn
        human_a = human_move(env)
        state, reward, done, info = env.step(human_a, -1)
        if done:
            env.render()
            print('Result:', info)
            break

if __name__ == '__main__':
    play()
    