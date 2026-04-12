---
name: environment-manager
description: Manage and detail the environment tasks for Gin Rummy, Liar's Dice, and Leduc Poker. Use when configuring routing, MCTS opponents, or reward shaping.
license: MIT
compatibility: Requires an agent with MCP sequential thinking server.
allowed-tools: mcp-server-sequential-thinking Bash(git:*)
metadata:
  version: "1.0"
---

# Environment Task Management Skill

This skill provides the structure, configuration details, and directives for maintaining the server-based game environments in this repository.

## Core Directives (MCP Thinking & Execution)
When adjusting the environments, debugging configurations, or modifying reward shaping logic, **you must use your MCP sequential thinking server** (`mcp-server-sequential-thinking` or internal chain-of-thought blocks) to plan the mathematical impacts of your changes **before** generating any code. Map out curriculum dependencies and probability distributions explicitly.

## Project Structure Overview

```text
project-root/
├── scripts/
│   ├── dockerfiles/                            # Core Docker image configurations
│   ├── affinetes/                              # Submodule: Backend environment servers
│   ├── open_spiel/                             # Submodule: OpenSpiel game definitions
│   ├── train_grpo_env.py                       # Main GRPO training execution loop
│   ├── gin_rummy_environment_function.py       # Gin Rummy Env Task
│   ├── liars_dice_environment_function.py      # Liar's Dice Env Task
│   └── leduc_poker_environment_function.py     # Leduc Poker Env Task
```

## Detailed Environment Configurations

### 1. Curriculum Scheduler (`CurriculumScheduler`)
All environment functions use a `CurriculumScheduler` class to incrementally scale the difficulty during training. It controls:
- `max_turn`: Progressively extending the length of the allowed game (e.g., fast checkmate scenarios to full-length games).
- `mcts_sims`: Linearly increasing the strength of the backend opponent.
- `hint_prob`: Fading out strategy guide prompts from 50% appearance to 0%.

### 2. Gin Rummy (`gin_rummy_environment_function.py`)
- **MCTS Opponent**: Configured for target MCTS(50, 1) simulations.
- **Reward Shaping**: 
  - `DEADWOOD_WEIGHT = 0.4`
  - Highly detailed shaping rewards that track the Bayesian estimate of the agent's hand deadwood. Bonuses (`DRAW_UPCARD_BONUS`) for selecting valid meld cards.
- **Strategy Hints**: Injected text instructing the agent on proper upcard evaluation and safe discarding using action index integers.

### 3. Liar's Dice (`liars_dice_environment_function.py`)
- **MCTS Opponent**: Highly aggressive MCTS(225, 1).
- **Curriculum**: Progresses from `TURN=2` (one claim/challenge exchange) to a cap of `20`.
- **Reward Shaping**: 
  - `PASS_MISSED_CHALLENGE_PENALTY`: -0.04 tuning for balancing bluff detection against an artificially strong opponent making legitimate high bids.
  - Plausibility penalties limit the model's reward when submitting mathematically outrageous bids.

### 4. Leduc Poker (`leduc_poker_environment_function.py`)
- **MCTS Opponent**: Progressive MCTS(10, 1) ramping to MCTS(50, 1).
- **Curriculum**: Game length caps at strictly 8 bounds. 
- **Reward Shaping**: 
  - Win/Loss terminal values multiplied up to +/- 30.
  - Mild penalties (-3.0) for poor folds and small bonuses (1.5) for preserving highest card strength in R1 without prematurely folding.

## Backend Interaction Flow
1. **Initialize (`_ensure_initialized`)**: Reads external URLs and pings HTTP `/reset` arrays initializing concurrent MCTS environments based on `MCTS_CONFIG`.
2. **Rollout Extractor**: Strips reasoning wrappers (`<thought>`, `<thinking>`) to isolate the strict action index from completions.
3. **HTTP Step**: Pushes action indexes asynchronously via `/step` payload to return new observation strings.