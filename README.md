## snake game engine that learns with iterations.

reward system:
* +10 for positive decision
* -10 for negative decision
*   0 for all other decision


11 input features -> 256 (Pytorch FP)->  3 Q-values

- (11, 256, 3)

+ `game.py`:Game loop, rendering and state management
+ `agent.py`:Agent decision-making and training
+ `model.py`:Neural network architecture

uv - package manager