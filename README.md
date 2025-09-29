# Tic-Tac-Toe
A machine learning "Tic Tac Toe Game " -> 

## README.md (short version)

```markdown
# Tic-Tac-Toe RL (Q-Learning)

Simple project that trains a Q-learning agent to play Tic-Tac-Toe.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

This will create `models/qtable.pkl`.

## Play

```bash
python -m src.play
```

## Notes
- Q-learning is tabular and will produce a Q-table you can inspect.
- Tweak `episodes`, `alpha`, `gamma`, `epsilon` in `src/train.py`.
```

---
