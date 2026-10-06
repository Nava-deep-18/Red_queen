# RED QUEEN AI ARENA: Game Theory Specification

## 1. Players
- **Player 1 (Attacker 🔴):** Goal is to reach the target cell.
- **Player 2 (Defender 🔵):** Goal is to catch (block/intercept) the Attacker before it reaches the target.

## 2. State Space
The state is represented as a fully observable grid or numeric vector containing:
- Position of the Attacker (x, y)
- Position of the Defender (x, y)
- Position of the Target (x, y)
- Positions of obstacles (if any)

This can be flattened into a fixed-length numeric vector suitable for a neural network input.

## 3. Action Space
Both players have a discrete action space of size 5:
- `0`: UP
- `1`: DOWN
- `2`: LEFT
- `3`: RIGHT
- `4`: STAY

## 4. Payoff Structure (Zero-Sum Game)
The game is strictly zero-sum, meaning `Payoff(Attacker) + Payoff(Defender) = 0` for all terminal outcomes.

| Outcome | Attacker Payoff | Defender Payoff |
| :--- | :--- | :--- |
| Attacker reaches target | +10 | -10 |
| Defender catches Attacker | -10 | +10 |
| Episode times out (no winner) | 0 | 0 |
| Invalid move (optional penalty) | -1 (optional) | +1 (optional) |

*Note: For the basic setup, step penalties might be added to encourage faster resolution (e.g., -0.1 per step for the attacker to encourage speed, +0.1 for the defender for surviving).*

## 5. Dynamics and Information
- **Turn Structure:** Simultaneous moves. Both agents choose their action without knowing the other's action for the current step.
- **Information:** Perfect information. Both agents know the exact state of the board.
- **Termination:** The episode ends when the Attacker reaches the target, the Defender catches the Attacker (they occupy the same cell), or a maximum number of steps (timeout) is reached.
