# Environment Functions

This reference focuses on the active training files under `scripts/envs/`.

## Shared Pattern Across Environment Files

Most env files follow the same structure:

1. Module-level constants for task metadata, curriculum knobs, and shaping coefficients (when present).
2. Observation formatting and action parsing helpers.
3. Optional parser/state dataclasses or reward calculators.
4. A `_curriculum_factory(args) -> CurriculumScheduler` helper that returns a scheduler (usually a subclass of the `CurriculumScheduler` base in `scripts/envs/shared_env.py`).
5. One or more rollout functions that:
   - initialize environment server connections (via `init_env_pool` from `shared_env`)
   - call `/reset`
   - call `generate_rollout_completions(...)`
   - parse the model action
   - call `/step`
   - compute shaping rewards
   - return rollout artifacts
6. `rollout_reward_func(completions, **kwargs)` that forwards environment rewards to TRL (typically re-exported from `shared_env`).

Public rollout functions are re-exported from `scripts/envs/__init__.py` with game-prefixed names (e.g., `gin_rummy_rollout_full_prompt_and_completion_parallelized_curriculum`).

## Tournament MCTS Rule

MCTS sim counts are fixed per env; there is **no progressive warmup** in the refactored path. The shared `CurriculumScheduler` has no `get_mcts_sims()` method.

| Env file | `mcts_max_simulations` | `mcts_num_rollouts` | Source |
|---|---|---|---|
| `scripts/envs/gin_rummy_env.py` | 25 | 1 | set inside `_ensure_initialized()` |
| `scripts/envs/gin_rummy_opponent_modeling.py` | 25 | 1 | module-level `_MCTS_SIMS = 25` |
| `scripts/envs/liar_dice_env.py` | 225 | 1 | set inside `_ensure_initialized()` |
| `scripts/envs/leduc_poker_env.py` | 50 | 1 | set inside `_ensure_initialized()` |
| `scripts/envs/goof_spiel_env.py` | — | — | no MCTS opponent |
| `scripts/envs/alf_world_env.py` | — | — | no MCTS opponent |

## MCTS Engine Details (from `open_spiel/open_spiel/algorithms/mcts.h` + `affinetes/environments/openspiel/env.py`)

Important when tuning strategy hints and understanding opponent behavior:

- **Evaluator**: `SafeRandomRolloutEvaluator(n_rollouts=n_rollouts)` — random playouts (NOT neural net priors).
- **n_rollouts=1** for all games that run MCTS — single random rollout per leaf node, high evaluation variance.
- **UCT exploration constant**: `uct_c = 1.414` (√2) — standard exploration-exploitation trade-off.
- **Action selection**: Highest visit count after all simulations (not highest value).
- **Hidden information**: Gin Rummy uses IS-MCTS which resamples opponent's unknown hand each simulation.
- **No learning**: MCTS does not adapt to opponent history or bluff patterns — each game is evaluated fresh.
- **Config override**: Training payload (`mcts_max_simulations`, `mcts_num_rollouts`) overrides agent-file defaults like `(3000,200)`.

Key weaknesses exploitable by the LLM agent:

- n_rollouts=1 → high variance in evaluation, especially in the 40-60% probability range.
- No Nash equilibrium convergence at low sim counts → LLM playing Nash strategy beats MCTS pure strategy.
- No bluff history tracking → consistent bluffing patterns are safe and undetected.
- IS-MCTS with 25 sims in gin → very limited hand sampling (~25 of millions of possible hands).

## `scripts/envs/shared_env.py`

The shared layer under every game file.

- `GAMES_TO_TASK_ID_RANGE: dict[str, tuple[int, int]]`
  Master task-id-range table (superset of what `_REGISTRY` actively routes).
- `CurriculumScheduler`
  Base class subclassed by each game's `_curriculum_factory`. Public methods:
  - `get_max_turn() -> int`
  - `get_hint_prob() -> float`
  - `step(num_rollouts: int = 1) -> None`
  - `get_status() -> dict`
- `init_env_pool(reset_payload, reset_endpoint="reset", lock_per_server=False) -> tuple[int, list[dict], int, ThreadPoolExecutor, Semaphore]`
  Initializes the environment server pool once per process. Reads `LOCAL_RANK` and `ENVIRONMENT_SERVER_URLS`.
- `rollout_reward_func(completions, **kwargs) -> list[float]`
  Generic passthrough; re-exported under game-prefixed names by each env file.

## `scripts/envs/goof_spiel_env.py`

### Module constants

- `_SELECTED_GAME = "goofspiel"`
- `_MAX_EPISODE_TOKENS = 16384`
- `_MAX_PROMPT_LEN = 4225`
- `_TIMEOUT = 2400`
- `_STRATEGY_REWARD_WEIGHT = 0.5` — full-prompt shaping weight
- `_STEP_STRATEGY_REWARD = 0.1` — full-prompt per-step reward
- `_STRATEGY_REWARD = 1.0` — last-prompt shaping reward
- `_INVALID_PENALTY = -0.1`

### Curriculum

Hint curriculum active at `initial_hint_prob=0.75`, `final_hint_prob=0.0` (set inside `_curriculum_factory`).

### Helpers

- `extract_and_format_observation(obs_text)`
- `extract_prize_card(obs_text)`
- `extract_bid_from_action(action_text, obs_text)`
- `get_hand_cards(observation_text, player_id=0)`
- `remove_reasoning_tags(text)`

### Rollout entry points

- `rollout_first_prompt_and_completion(...)` — unique-to-goofspiel (no action mask)
- `rollout_full_prompt_and_completion_parallelized_curriculum(...)` — full-prompt with action mask
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)` — last-prompt
- `rollout_reward_func(completions, **kwargs)`

Public prefixed exports in `scripts/envs/__init__.py`: `goof_spiel_rollout_first_prompt_and_completion`, `goof_spiel_rollout_full_prompt_and_completion_parallelized_curriculum`, `goof_spiel_rollout_last_prompt_and_completion_parallelized_curriculum`, `goof_spiel_rollout_reward_func`.

Current role:

- Goofspiel is the simplest active env and is useful as the easiest reference for the general rollout pattern.

## `scripts/envs/gin_rummy_env.py`

### Module constants

No reward/shaping constants at module scope — shaping lives inside the `RewardCalculator` class body. Module-level items:

- `_SELECTED_GAME = "gin_rummy"`
- `_MAX_EPISODE_TOKENS = 16384`
- `_MAX_PROMPT_LEN = 5000`
- `_TIMEOUT = 2400`
- `CARD_VALUES: dict[str, int]`
- `RANK_ORDER: list[str]`

### Curriculum

Hint curriculum active at `initial_hint_prob=0.5`, `final_hint_prob=0.0`. Final max turn set inside `_curriculum_factory`; initial max turn comes from trainer args (overridden per-mode in `env_configs.py`).

### Card and meld helpers

- `get_rank(card)`, `get_suit(card)`, `get_value(card)`
- `find_potential_runs(hand, additional_card=None)`
- `count_complete_runs(hand)`
- `find_all_melds(hand)`
- `compute_optimal_deadwood(hand)`
- `meld_potential(upcard, hand)`

### State and classes

- `GameState` (dataclass)
  Hand / deadwood / phase / knock card / upcard / stock size / discard pile / player id, with helper methods like `total_hand_value()`, `can_knock()`, `count_pairs()`, `count_runs()`, `count_sets()`.
- `RewardCalculator`
  - `calculate_step_reward(...)`
  - `calculate_episode_reward(...)`
  - shaping constants live as class attributes / locals rather than module-scope constants.

### Parsing helpers

- `extract_and_format_observation(obs_text)`
- `parse_hand_from_observation(observation)`
- `parse_discard_pile(observation)`
- `parse_game_state(observation)`
- `remove_reasoning_tags(text)`
- `extract_action_id(completion_text)`

Action ids `52`, `53`, and `54` remain semantically important (draw-upcard / draw-stock / knock sentinels) in prompts and shaping logic.

### Rollout entry points

- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)` — re-exported from `shared_env`

Public prefixed exports: `gin_rummy_rollout_full_prompt_and_completion_parallelized_curriculum`, `gin_rummy_rollout_last_prompt_and_completion_parallelized_curriculum`, `gin_rummy_rollout_reward_func`.

## `scripts/envs/gin_rummy_opponent_modeling.py`

Variant of Gin Rummy with Bayesian opponent modeling. Routed via env name `gin_rummy_opponent_modeling` in `_REGISTRY`, or via `_VARIANT_OVERRIDES` to swap in for `gin_rummy` wholesale.

### Module constants

- `_SELECTED_GAME = "gin_rummy"`
- `_MAX_EPISODE_TOKENS = 16384`
- `_MAX_PROMPT_LEN = 16384 - 256`
- `_TIMEOUT = 2400`
- `_MCTS_SIMS = 25`

Reward coefficients (lines 42-54):

- `TERMINAL_WIN_REWARD = 1.0`
- `TERMINAL_LOSS_REWARD = -1.0`
- `GIN_BONUS = 0.25`
- `KNOCK_BONUS = 0.1`
- `DEADWOOD_WEIGHT = 0.5`
- `INVALID_PENALTY = -0.1`
- `INVALID_TOTAL_CLIP = -0.3`
- `TERMINAL_REWARD_CLIP = 1.0`
- `SAFE_DISCARD_BONUS = 0.02`
- `DANGEROUS_DISCARD_PENALTY = 0.02`
- `DRAW_UPCARD_BONUS = 0.03`
- `DRAW_UPCARD_PENALTY = 0.02`

### Bayesian opponent classes

- `DeadCardTracker` (line 245)
  "Tracks discarded cards and identifies layoff candidates." Important methods:
  - `update_from_discard_pile(...)`
  - `update_from_observation(...)`
  - `get_dead_cards()`
  - `is_dead(card)`
  - `get_layoff_candidates(hand, discard_pile)`
  - `summary(hand)`
- `BayesianOpponentModel` (line 331)
  "Infers opponent's meld direction from discard pile deltas." Lightweight rank/suit heat model for dangerous vs safe discards.
- `BayesianOpponentHandModel` (line 419)
  "Tracks P(card ∈ opponent_hand | all observations) via Bayesian updates." Estimates knock risk and likely meld cards.

### Curriculum

Hint curriculum `initial_hint_prob=0.5`, `final_hint_prob=0.0`. `_HINT_PROMPT` module constant contains the phase-by-phase decision guide plus the IS-MCTS exploitation section.

### Rollout entry points

- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)`

These are not re-exported with a `gin_rummy_opponent_modeling_` prefix in `__init__.py`; reach them via `get_env_config("gin_rummy_opponent_modeling")`.

## `scripts/envs/liar_dice_env.py`

### Module constants

- `_SELECTED_GAME = "liars_dice"`
- `_MAX_EPISODE_TOKENS = 16384`
- `_MAX_PROMPT_LEN = 5000`
- `_TIMEOUT = 2400`

Risky-play bonus scheme (lines 32-39):

- `BLUFF_PROB_THRESHOLD = 0.35`
- `RISKY_LIAR_PROB_MIN = 0.35`
- `RISKY_LIAR_PROB_MAX = 0.60`
- `BLUFF_WIN_BONUS = 0.5`
- `RISKY_LIAR_WIN_BONUS = 0.5`
- `RISKY_BONUS_MAX_COUNT = 2`
- `SHUFFLE_PROB = 0.5`
- `NORMALIZE_REWARDS = False`

The pre-refactor constants (`PASS_MISSED_CHALLENGE_PENALTY`, `BID_PLAUSIBILITY_BONUS`, `BID_PLAUSIBILITY_PENALTY`, `SHAPING_REWARD_CLIP`, `TERMINAL_REWARD_CLIP`, `INVALID_ACTION_PENALTY`, `CURRICULUM_INITIAL_MCTS_SIMS`, `CURRICULUM_FINAL_MCTS_SIMS`) are **gone** from module scope. Do not reintroduce them without checking the new shaping path.

Env-var overrides (`LIARS_DICE_INITIAL_HINT_PROB`, `LIARS_DICE_FINAL_HINT_PROB`, `LIARS_DICE_RULESET`, `LIARS_DICE_FINAL_MAX_TURN`, `EPISODE_TRACE_*`) were also removed. Configure hint probabilities in `_curriculum_factory` instead.

### Data classes

- `Bid` (dataclass) — quantity + face value.
- `Action` (dataclass) — id, label, optional `Bid`, probability; properties `is_liar`, `aggressiveness`, `score`.
- `GameState` (dataclass) — dice, bid state, list of legal actions; properties `liar_action`, `bid_actions`.

### Parsing and shaping helpers

- `parse_game_state(messages) -> GameState`
- `bid_probability(bid: Bid | None, state: GameState) -> float` — binomial tail probability that a bid is truthful.
- Various `_extract_*`, `_score_*` helpers that feed the `RewardCalculator`.
- `RewardCalculator` — shaped reward computation, honors the `NORMALIZE_REWARDS` flag.

### Curriculum

- `initial_hint_prob = 0.5`, `final_hint_prob = 0.0` (set inside `_curriculum_factory`).
- Final max turn: 15.
- Initial max turn comes from trainer args; overridden per-mode to `1` in all three `ModeConfig` slots for `liars_dice` with `rollouts_per_stage=2048`.

### Rollout entry points

- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)`

Public prefixed exports: `liar_dice_rollout_full_prompt_and_completion_parallelized_curriculum`, `liar_dice_rollout_last_prompt_and_completion_parallelized_curriculum`, `liar_dice_rollout_reward_func`.

Env-level registry knobs: `num_generations=8`, `temperature=2.0`, `top_k=5`.

## `scripts/envs/leduc_poker_env.py`

### Module constants

- `_INVALID_PENALTY = -0.1`
- `_MAX_TURNS = 10`
- `_BASE_SYSTEM_PROMPT` — game rules prompt
- `_HINT_PROMPT` — Nash equilibrium strategy guide + MCTS exploitation section

Old `CURRICULUM_*` module constants (`CURRICULUM_INITIAL_TURN`, `CURRICULUM_FINAL_TURN`, `CURRICULUM_ROLLOUTS_PER_STAGE`, `CURRICULUM_WARMUP_ROLLOUTS`, `CURRICULUM_INITIAL_HINT_PROB`, `CURRICULUM_FINAL_HINT_PROB`, `CURRICULUM_INITIAL_MCTS_SIMS`, `CURRICULUM_FINAL_MCTS_SIMS`) are gone from module scope — curriculum knobs are supplied via `_curriculum_factory(args)` and the `ModeConfig` overrides.

### Curriculum

- `initial_hint_prob = 0.75`, `final_hint_prob = 0.0`.
- No progressive MCTS warmup; `_ensure_initialized()` sets `mcts_max_simulations=50`, `mcts_num_rollouts=1`.

### Helpers

- `remove_reasoning_tags(text)`
- `parse_action(completion_text)`
- `extract_and_format_observation(obs_text)`
- `parse_poker_state(obs)` / `GameState` dataclass

### Reward logic

- `RewardCalculator` — shaped rewards live inside the class body; includes pair bonus, raise-with-pair bonus, high-card strength bonus, early weak-fold penalty, pot-growth reward, and scaled terminal payoff. Exact coefficients are class attributes or locals rather than module constants.

### Rollout entry points

- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)`

Public prefixed exports: `leduc_poker_rollout_full_prompt_and_completion_parallelized_curriculum`, `leduc_poker_rollout_last_prompt_and_completion_parallelized_curriculum`, `leduc_poker_rollout_reward_func`.

Env-level registry knobs: `num_generations=8`, `temperature=2.0`, `top_k=5`. No per-mode overrides in the current registry.

## `scripts/envs/alf_world_env.py`

AlfWorld household task environment. No curriculum, no MCTS opponent; registered in `_REGISTRY` with `curriculum_factory=None`.

### Module constants

- `_MAX_EPISODE_TOKENS = 16384`
- `_MAX_PROMPT_LEN = 24576`
- `_TIMEOUT = 2400`
- `_CONVERSATION_START: list[dict]` — system + assistant messages for household interaction

### Rollout entry points

- `alfworld_rollout_first_prompt_and_completion_parallelized(...)`
- `alfworld_rollout_full_prompt_and_completion_parallelized(...)`
- `alfworld_rollout_reward_func(completions, **kwargs)`

No "last prompt" variant and no prefixed renames — these are already the public names in `scripts/envs/__init__.py`.

In the registry, `rollout_last` is assigned to `alfworld_rollout_first_prompt_and_completion_parallelized` so the two non-masked modes still have a callable.

## What To Compare When Debugging

- Wrong action ids:
  Compare `remove_reasoning_tags`, action parser, legal-action block format in `scripts/envs/<game>_env.py`, and backend agent formatting in `affinetes/environments/openspiel/agents/`.
- Bad reward learning:
  Compare the env file's `RewardCalculator`, parser/state extraction, and how `env_rewards` are returned to TRL through the game-prefixed `rollout_reward_func` export.
- Curriculum not progressing:
  Compare the env's `_curriculum_factory` and `CurriculumScheduler.step(...)` updates against `trainer.args.rollouts_per_stage`/`initial_max_turn`, plus any `ModeConfig` overrides in `scripts/envs/env_configs.py`.
- Mask misalignment:
  Compare the full-rollout token accumulation with `ActionMaskedGRPOTrainer` expectations in `scripts/train_grpo_env.py`; only full-prompt mode uses the action-masked trainer.
- Wrong variant selected:
  Check `_VARIANT_OVERRIDES` in `scripts/envs/env_configs.py`. An active entry there can silently redirect `gin_rummy` → `gin_rummy_opponent_modeling`.
