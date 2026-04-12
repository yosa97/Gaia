# Environment Functions

This reference focuses on the active environment-training files in `scripts/`.

## Shared Pattern Across Environment Files

Most environment files follow the same structure:

1. Constants for task-id ranges, MCTS config, curriculum knobs, and shaping coefficients.
2. Observation formatting and action parsing helpers.
3. Optional parser/state dataclasses or reward calculators.
4. A curriculum scheduler that tracks rollout count or optimizer-step progress.
5. One or more rollout functions that:
   - initialize environment server connections
   - call `/reset`
   - call `generate_rollout_completions(...)`
   - parse the model action
   - call `/step`
   - compute shaping rewards
   - return rollout artifacts
6. `rollout_reward_func(...)` that forwards environment rewards to TRL.

## Tournament MCTS Rule

Unless a task explicitly asks for a different opponent setting, keep the tournament default opponent as MCTS with these target settings:

- Gin Rummy: `MCTS(50,1)` — IS-MCTS (information set sampling for hidden cards), progressive warmup from 10 sims
- Liars Dice: `MCTS(225,1)` — classic UCT MCTS, highest sim count for strongest opponent
- Leduc Poker: `MCTS(50,1)` — classic UCT MCTS, progressive warmup from 10 sims

When curriculum is present, weaker warmup settings can ramp toward these targets, but these are the steady-state defaults that should be preserved.

## MCTS Engine Details (from `open_spiel/open_spiel/algorithms/mcts.h` + `affinetes/environments/openspiel/env.py`)

These facts are important when tuning strategy hints and understanding opponent behavior:

- **Evaluator**: `SafeRandomRolloutEvaluator(n_rollouts=n_rollouts)` — uses random playouts (NOT neural net priors)
- **n_rollouts=1** for all 3 active games — a single random rollout per leaf node, causing very high evaluation variance
- **UCT exploration constant**: `uct_c = 1.414` (√2) — standard exploration-exploitation trade-off
- **Action selection**: Highest visit count after all simulations (not highest value)
- **Hidden information**: Gin Rummy uses IS-MCTS which resamples opponent's unknown hand each simulation
- **No learning**: MCTS does not adapt to opponent history or bluff patterns — each game is evaluated fresh
- **Config override**: Training payload (`mcts_max_simulations`, `mcts_num_rollouts`) overrides agent-file defaults like `(3000,200)`

Key weaknesses exploitable by the LLM agent:
- n_rollouts=1 → **high variance** in evaluation, especially in the 40-60% probability range
- No Nash equilibrium convergence at low sim counts → LLM playing Nash strategy beats MCTS pure strategy
- No bluff history tracking → consistent bluffing patterns are safe and undetected
- IS-MCTS with 50 sims → very limited hand sampling for Gin Rummy (~50 of millions of possible hands)

## `scripts/goof_spiel_environment_function.py`

Main helpers:

- `extract_and_format_observation(obs_text)`
  Rebuilds the legal-action block from the player's hand so the observation matches evaluation-style formatting.
- `extract_prize_card(obs_text)`
  Pulls the current prize card from the observation.
- `extract_bid_from_action(action_text, obs_text)`
  Converts action id back into bid value.
- `get_hand_cards(observation_text, player_id=0)`
  Parses the remaining hand cards.
- `remove_reasoning_tags(text)`
  Strips `<think>`-style tags.
- `CurriculumScheduler`
  Tracks `max_turn` and hint probability from total rollout count.

Rollout functions:

- `rollout_first_prompt_and_completion(...)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)`

Current role:

- Goofspiel is the simplest environment path in this workspace and is useful as the easiest reference for the general rollout pattern.

## `scripts/gin_rummy_environment_function.py`

### Core constants

- `CARD_VALUES`
- `RANK_ORDER`
- `MCTS_CONFIG` — `{"opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}`
- `TERMINAL_WIN_REWARD = 1.0`
- `TERMINAL_LOSS_REWARD = -1.0`
- `GIN_BONUS = 0.25` — bonus for 0-deadwood win
- `KNOCK_BONUS = 0.1` — bonus for winning via knock
- `DEADWOOD_WEIGHT = 0.4` — reduced from 0.5 for MCTS(50,1); terminal signal dominates more
- `INVALID_PENALTY = -0.1`
- `INVALID_TOTAL_CLIP = -0.3`
- `TERMINAL_REWARD_CLIP = 1.0`
- `SAFE_DISCARD_BONUS = 0.02`
- `DANGEROUS_DISCARD_PENALTY = 0.02`
- `DRAW_UPCARD_BONUS = 0.03`
- `DRAW_UPCARD_PENALTY = 0.02`
- `CURRICULUM_INITIAL_MCTS_SIMS = 10` — progressive MCTS warmup start
- `CURRICULUM_FINAL_MCTS_SIMS = 50` — matches MCTS_CONFIG target
- `_HINT_PROMPT` — module-level strategy guide injected into early episodes (50%→0% via hint curriculum)

### Card and meld helpers

- `get_rank(card)`
- `get_suit(card)`
- `get_value(card)`
- `find_potential_runs(hand, additional_card=None)`
- `count_complete_runs(hand)`
- `find_all_melds(hand)`
- `compute_optimal_deadwood(hand)`
- `meld_potential(upcard, hand)`
- `draw_ucb_shaping(current_state, chosen_action_id)`

These functions implement the meld/deadwood logic that makes gin-rummy shaping more precise than a plain win/loss reward.

### State and inference helpers

- `GameState`
  Fields:
  - `hand`
  - `deadwood`
  - `phase`
  - `knock_card`
  - `upcard`
  - `stock_size`
  - `discard_pile`
  - `player_id`

  Methods:
  - `total_hand_value()`
  - `num_high_cards()`
  - `can_knock()`
  - `count_pairs()`
  - `count_sets()`
  - `count_runs()`
  - `count_potential_runs()`

- `DeadCardTracker`
  Tracks discard history and heuristic layoff candidates.

  Important methods:
  - `update_from_discard_pile(...)`
  - `update_from_observation(...)`
  - `get_dead_cards()`
  - `is_dead(card)`
  - `get_layoff_candidates(hand, discard_pile)`
  - `summary(hand)`

- `BayesianOpponentModel`
  Lightweight rank/suit heat model for dangerous versus safe discards.

  Important methods:
  - `_update_heat(card, weight)`
  - `update_on_opponent_draw(drawn_card)`
  - `update_on_opponent_discard(discarded_card)`
  - `update_from_discard_pile_delta(prev_discard_pile, curr_discard_pile)`
  - `is_dangerous_discard(card)`
  - `is_safe_discard(card)`
  - `get_danger_cards(hand)`
  - `get_safe_cards(hand)`
  - `summary(hand)`

- `BayesianOpponentHandModel`
  Posterior-style estimate of likely opponent holdings and knock risk.

  Important methods:
  - `initialize(our_hand, discard_pile)`
  - `update_opp_drew_upcard(upcard)`
  - `update_opp_drew_stock()`
  - `update_opp_discarded(card)`
  - `estimated_opponent_hand(top_n=10)`
  - `knock_risk()`
  - `likely_meld_cards()`
  - `summary(hand)`

### Parsing helpers

- `extract_and_format_observation(obs_text)`
- `parse_hand_from_observation(observation)`
- `parse_discard_pile(observation)`
- `parse_game_state(observation)`
- `remove_reasoning_tags(text)`
- `extract_action_id(completion_text)`

Important current assumptions:

- Action ids `52`, `53`, and `54` are semantically important in prompts and shaping logic.
- Observation parsing depends on backend formatting such as `You are Player X`, `Deadwood=`, `Phase:`, `Knock card:`, `Stock size:`, and `Discard pile:`.

### Reward and curriculum

- `RewardCalculator`
  Important methods:
  - `calculate_step_reward(...)`
  - `compute_discard_safety(states)`
  - `calculate_episode_reward(...)`

- `CurriculumScheduler`
  Important methods:
  - `get_max_turn()`
  - `get_hint_prob(optimizer_step=None)`
  - `get_mcts_sims(optimizer_step=None)`
  - `step(num_rollouts=1)`
  - `get_status(optimizer_step=None)`

Current behavior:

- Curriculum separately ramps turns, hint probability, and MCTS simulations.
- Hint decay uses optimizer steps (`hint_decay_optimizer_steps=100`).
- MCTS difficulty ramps from `CURRICULUM_INITIAL_MCTS_SIMS=10` to `CURRICULUM_FINAL_MCTS_SIMS=50`.
- Hint probability starts at `0.5` and decays to `0.0` — both CurriculumScheduler instances pass these explicitly.
- `_HINT_PROMPT` contains phase-by-phase decision guide + IS-MCTS exploitation section.

### Rollout entry points

- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
  Returns only the latest prompt/completion artifacts for the episode.
- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
  Returns the full episode prompt/completion stream and action mask.
- `rollout_reward_func(completions, **kwargs)`

Current rollout characteristics:

- Initializes server pool lazily and caches it as function attributes.
- Uses a semaphore to serialize generation calls.
- Augments later observations with dead-card and Bayesian summaries.
- Calls `/reset` and `/step` against `ENVIRONMENT_SERVER_URLS`.
- Computes final training reward from deadwood improvement, terminal result, and invalid-action penalties.

## `scripts/liars_dice_environment_function.py`

### Core constants and variants

- `GAME_TO_TASK_ID_RANGE`
- `SELECTED_GAME = "liars_dice"`
- `REQUEST_TIMEOUT_SECONDS = 2400`
- `INIT_TIMEOUT_SECONDS = 300`
- `MAX_EPISODE_TOKENS = 16384`
- `MAX_PROMPT_LEN = 16384 - 512`
- `MCTS_CONFIG` — `{"opponent": "mcts", "mcts_max_simulations": 225, "mcts_num_rollouts": 1}`
- `CURRICULUM_INITIAL_TURN = 2` — fixed constant (trainer.args.initial_max_turn holds MCTS sim count, NOT turn count)
- `INVALID_ACTION_PENALTY = 0.10`
- `PASS_MISSED_CHALLENGE_PENALTY = 0.04` — reduced from 0.06 for MCTS(225,1) stronger opponent
- `BID_PLAUSIBILITY_BONUS = 0.03` — reduced from 0.04
- `BID_PLAUSIBILITY_PENALTY = 0.04`
- `SHAPING_REWARD_CLIP = 0.35` — reduced from 0.50; tighter clip so terminal dominates
- `TERMINAL_REWARD_CLIP = 1.00`
- `CURRICULUM_INITIAL_HINT_PROB = 0.5` — default for env var `LIARS_DICE_INITIAL_HINT_PROB`
- `CURRICULUM_FINAL_HINT_PROB = 0.0` — default for env var `LIARS_DICE_FINAL_HINT_PROB`
- `RULESET_CLASSIC = "classic"`
- `RULESET_LIARS_DIE = "liars_die"`
- `STRATEGY_TIPS_CLASSIC` — probability-anchored guide with MCTS exploitation section
- `STRATEGY_TIPS_LIARS_DIE` — single-die variant guide with MCTS exploitation section

The file supports both:

- classic multi-dice liar's dice (2 players × 5 dice = 10 total; 6s are wild)
- a single-die "liars_die" variant (FSICFR-style rank claims, Doubt vs Accept)

### Generic helpers

- `_is_truthy_env(value)`
- `_safe_float(value, default=0.0)`
- `_clamp(value, min_value, max_value)`
- `_ruleset_from_env()`
- `_detect_ruleset_from_observation(observation)`
- `resolve_ruleset(observation)`
- `extract_and_format_observation(obs_text)`
- `remove_reasoning_tags(text)`

### Logging and curriculum

- `EpisodeTraceLogger`
  JSONL tracer controlled by env vars such as `EPISODE_TRACE_ENABLED`, `EPISODE_TRACE_DIR`, `EPISODE_TRACE_MAX_TEXT_CHARS`, and `EPISODE_TRACE_SAMPLE_RATE`.
- `CurriculumScheduler`
  Controls turn budget and hint probability from total rollout count.

### Parsing and shaping helpers

- `_extract_legal_action_map(observation)`
- `_extract_bid_tuple(label_or_text)`
- `_extract_state_features(observation, ruleset=RULESET_CLASSIC)`
- `_extract_liars_die_state_features(observation)`
- `_liars_die_parse_action(label)`
- `_liars_die_compute_shaping(state, action_kind, claim_rank)`
- `_is_liar_label(label)`
- `_bid_rank(bid)`
- `_count_face_support(own_dice, target_face, wild_six_enabled)`
- `_binomial_tail_probability(num_trials, success_prob, min_successes)`
- `_estimate_bid_statistics(state_features, bid)`
- `_score_bid_plausibility(state_features, bid)`
- `_parse_action_id(completion_text, legal_action_map, ruleset=RULESET_CLASSIC)`
- `_score_challenge_decision(state_features, chose_liar, proposed_bid)`
- `_select_fallback_action(legal_action_map, state_features, ruleset=RULESET_CLASSIC)`
- `_extract_terminal_reward(step_block, observation_text)`

These are the core probability and decision-quality functions for liar's-dice shaping.

### Environment and rollout helpers

- `_build_env_pool(server_urls)`
- `_initialize_rollout_state(trainer)`
- `_reset_environment(env_endpoint, game_id, timeout)`
- `_step_environment(env_endpoint, episode_id, action_to_send, timeout)`
- `_last_prompt_fallback_result()`
- `_full_prompt_fallback_result()`
- `_execute_parallel_rollouts(prompts, executor, run_single_prompt, fallback_builder)`
- `_log_batch_statistics(list_results)`
- `_get_system_prompt(use_hints, ruleset=RULESET_CLASSIC)`
- `_rollout_parallelized_curriculum(prompts, trainer, include_action_mask)`

### Public rollout entry points

- `rollout_last_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_full_prompt_and_completion_parallelized_curriculum(...)`
- `rollout_reward_func(completions, **kwargs)`

Current rollout characteristics:

- Uses a module-level `_ROLLOUT_STATE` cache instead of function attributes.
- Can change ruleset via env var or observation auto-detection.
- Can emit full action masks for `ActionMaskedGRPOTrainer`.
- Contains fallback action logic when parsing fails.
- Separates classic bid plausibility from the single-die accept/doubt variant.
- Hint prob defaults: `CURRICULUM_INITIAL_HINT_PROB=0.5` and `CURRICULUM_FINAL_HINT_PROB=0.0` (env vars override).
- Init log prints: `[CURRICULUM] Initialized: turns {initial}→{final}, mcts_sims=225, hints 0.5→0.0`

## `scripts/leduc_poker_environment_function.py`

### Core constants

- `GAME_TO_TASK_ID_RANGE`
- `SELECTED_GAME = "leduc_poker"`
- `TIMEOUT = 2400`
- `ACTION_FOLD = "0"`, `ACTION_CALL = "1"`, `ACTION_RAISE = "2"`
- `MCTS_CONFIG` — `{"opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}`
- `CURRICULUM_INITIAL_TURN = 2` — start simple: one bet/response round
- `CURRICULUM_FINAL_TURN = 8` — full game length (2 rounds × up to 4 bets)
- `CURRICULUM_ROLLOUTS_PER_STAGE = 512` — 6 stages × 512 = 3072 rollouts to reach max
- `CURRICULUM_WARMUP_ROLLOUTS = 128`
- `CURRICULUM_INITIAL_HINT_PROB = 0.5` — 50% of early episodes include Nash strategy hints
- `CURRICULUM_FINAL_HINT_PROB = 0.0` — no hints by end of training
- `CURRICULUM_INITIAL_MCTS_SIMS = 10` — weaker opponent during early curriculum
- `CURRICULUM_FINAL_MCTS_SIMS = 50` — matches MCTS_CONFIG at full curriculum
- `_HINT_PROMPT` — Nash equilibrium strategy guide (Round 1/2 decisions + MCTS exploitation section)
- `_SYSTEM_PROMPT` — game rules prompt (Leduc Poker deck, actions, hand ranking)

### Helpers

- `CurriculumScheduler`
  Methods:
  - `get_max_turn()`
  - `get_hint_prob()`
  - `get_mcts_sims()`
  - `step(num_rollouts=1)`
- `remove_reasoning_tags(text)`
- `parse_action(completion_text)`
- `extract_and_format_observation(obs_text)`
- `parse_poker_state(obs)`

### Reward logic

- `RewardCalculator`
  Important methods:
  - `calculate_step_reward(prev_state, curr_state, action, is_invalid, is_terminal, final_env_reward=0.0)`
  - `calculate_discounted_return(rewards)`

Current shaping signals:

- pair bonus
- high-card strength bonus
- early weak-fold penalty
- pot growth reward
- scaled terminal payoff

### Rollout helpers and entry points

- `_ensure_initialized(fn, trainer)`
- `rollout_last_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30)`
- `rollout_full_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30)`
- `rollout_reward_func(completions, **kwargs)`

Current rollout characteristics:

- Uses function-attribute caches like the gin-rummy file.
- Ramps both turns and MCTS difficulty.
- Uses a short action space with raw ids `0`, `1`, and `2`.
- Full rollout builds `action_mask` and discounted return.

## What To Compare When Debugging

- Wrong action ids:
  Compare `remove_reasoning_tags`, action parser, legal-action block format, and backend agent formatting.
- Bad reward learning:
  Compare reward calculator, parser/state extraction, and how `env_rewards` are returned to TRL.
- Curriculum not progressing:
  Compare scheduler state updates with `trainer.args.rollouts_per_stage`, warmup knobs, and rollout cache initialization.
- Mask misalignment:
  Compare full-rollout token accumulation with `ActionMaskedGRPOTrainer` expectations in `scripts/train_grpo_env.py`.
