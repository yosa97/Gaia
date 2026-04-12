# Tournament Environments Project Instructions

## Project Structure

```text
project-root/
├── scripts/                                    # Core framework and training code
│   ├── dockerfiles/                            # Core Docker image configurations
│   ├── affinetes/                              # Backend environment code
│   │   └── environments/                       # Source game rule implementations
|   |      └── openspiel/                       # OpenSpiel game implementations
|   |           └── agents/                     # game agent ex: (gin_rummy.py, liars_dice_agent.py, leduc_poker_agent.py)
│   ├── grpo_env_config.py                      # GRPO training configurations
│   ├── train_grpo_env.py                       # Main GRPO training execution loop
│   │
│   │   # Environment Task Functions
│   │   # These interface with the game servers to handle rollouts, curriculum scheduling, 
│   │   # strategy hints, MCTS opponents, and reward shaping per game.
│   ├── gin_rummy_environment_function.py       # Gin Rummy: MCTS(50,1) config, Bayesian reward shaping, and deadwood evaluation.
│   ├── liars_dice_environment_function.py      # Liar's Dice: MCTS(225,1) config, bid plausibility tracking, and bluff penalties.
│   └── leduc_poker_environment_function.py     # Leduc Poker: MCTS(50,1) config, mixed strategy rewards, and progressive curriculum.
│
├── .claude/
│   └── skills/                                 # Claude Code skills
```

### Environment Tasks Overview
The primary focus of this repository is training agents via GRPO to optimally play turn-based game environments. The game interaction logic is completely isolated into the `*_environment_function.py` scripts. 

For each game environment, the task script manages:
1. **Curriculum Scheduling**: Progressively ramping up difficulty by extending the maximum allowed turns (`max_turns`), reducing strategy hints (`hint_prob`), and increasing the strength of the baseline programmatic opponent (`mcts_sims`).
2. **Server Communication**: Hitting external API endpoints (`/reset` and `/step`) to initialize games and register actions against the MCTS player.
3. **Reward Shaping**: Translating binary terminal outputs (Win/Loss) into dense internal rewards, calculating immediate payoffs for intermediate decisions (e.g., Pot Growth in Leduc Poker, Bid Plausibility in Liar's Dice, or Deadwood improvement in Gin Rummy).
4. **Action Parsing**: Cleaning `<thought>` reasoning tags before translating the LLM string outputs into actionable network states.