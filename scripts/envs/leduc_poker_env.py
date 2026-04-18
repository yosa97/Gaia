import functools
import os
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 4096
_TIMEOUT = 2400
_INVALID_PENALTY = -0.1
_MAX_TURNS = 10

# System prompts
_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits \u00d7 (num_players + 1) ranks. For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check (match current bet), "
    "Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Guide\n"
    "ROUND 1:\n"
    "- K in hand \u2192 Raise (strongest non-pair; builds pot for potential R2 pair)\n"
    "- Q in hand \u2192 Call (middle hand; wait to see public card)\n"
    "- J in hand \u2192 Call; fold if opponent raises twice (weakest hand, bad pot odds)\n\n"
    "ROUND 2 (public card now visible):\n"
    "- Public card SAME RANK as your card \u2192 PAIR \u2192 always Raise (dominant hand)\n"
    "- No pair + K \u2192 Call opponent raises (K beats Q and J without pair)\n"
    "- No pair + Q \u2192 Call if opponent only called; Fold to raises\n"
    "- No pair + J \u2192 Fold to any Raise (weakest non-pair)\n\n"
    "READING OPPONENT:\n"
    "- Opponent raised R1 then checked R2 \u2192 likely missed pair (caught bluffing)\n"
    "- Opponent raised both rounds \u2192 likely has a pair; be cautious without one\n"
    "- Opponent folded to your raise \u2192 bet was credible; note their threshold\n\n"
    "EXPLOITING THE MCTS OPPONENT (1 random rollout per node):\n"
    "- Leduc Poker has only 936 total information states; at this sim budget MCTS covers only a fraction per decision\n"
    "- MCTS uses random rollouts (not Nash equilibrium) \u2192 it underestimates bluffing value\n"
    "- Random rollouts from any position win ~1/3 of the time \u2192 MCTS sees all positions as similar\n"
    "- Play Nash equilibrium (the strategy guide above) \u2014 it ALWAYS outperforms MCTS pure strategy\n"
    "- Key exploit: MCTS is overly passive with J \u2014 raise with K/Q more than MCTS expects\n"
    "- Key exploit: MCTS folds too rarely vs aggressive raises \u2014 raise more with pairs in R2\n"
    "- MCTS cannot adapt its strategy based on your betting history \u2014 consistent patterns are safe\n"
)


# ---------------------------------------------------------------------------
# Game state dataclass and parser
# ---------------------------------------------------------------------------

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


@dataclass
class GameState:
    """
    Structured representation of a Leduc Poker observation.

    All fields are parsed directly from the observation text.  Derived
    properties compute common strategy quantities so reward-shaping code
    can stay readable.

    Betting history notes
    ---------------------
    P0 always acts first in each round.  The betting lists record every
    action taken so far in that round, interleaved in turn order:
      r1_betting = ["Raise", "Call"]  → P0 raised, P1 called
      r2_betting = ["Call", "Raise"]  → P0 checked, P1 raised

    When it is our turn, the last entry in the current-round list is always
    the most recent *opponent* action (because the lists only contain
    completed actions — ours is not recorded until after we respond).
    """
    player_id:         int               # 0 or 1
    private_card:      str               # e.g. "K♠"
    private_card_rank: int               # J=1, Q=2, K=3
    public_card:       "str | None"      # None in round 1
    public_card_rank:  "int | None"
    has_pair:          bool              # True when Hand: Pair appears
    round:             int               # 1 or 2
    pot:               int
    our_chips:         int
    opp_chips:         int
    r1_betting:        list[str]         # e.g. ["Raise", "Call"]
    r2_betting:        list[str]         # empty until round 2
    legal_actions:     dict[int, str]    # {0: "Fold", 1: "Call", 2: "Raise"}

    # ---- derived properties ------------------------------------------------

    @property
    def our_invested(self) -> int:
        """Chips we have put into the pot so far (= 100 - our_chips)."""
        return 100 - self.our_chips

    @property
    def opp_invested(self) -> int:
        """Chips opponent has put into the pot so far."""
        return 100 - self.opp_chips

    @property
    def current_round_betting(self) -> list[str]:
        return self.r1_betting if self.round == 1 else self.r2_betting

    @property
    def raises_this_round(self) -> int:
        """Total raises already taken in the current round (by both players)."""
        return self.current_round_betting.count("Raise")

    @property
    def opp_last_action(self) -> "str | None":
        """
        The most recent action the opponent took in the current round, or None
        if the opponent has not yet acted this round.

        Because P0 always leads, the last entry in the current-round betting
        list is always the opponent's most recent move (since it's our turn).
        """
        betting = self.current_round_betting
        return betting[-1] if betting else None

    @property
    def opp_raised_this_round(self) -> bool:
        return "Raise" in self.current_round_betting

    @property
    def can_raise(self) -> bool:
        return 2 in self.legal_actions

    @property
    def can_fold(self) -> bool:
        return 0 in self.legal_actions

    @property
    def hand_strength(self) -> int:
        """
        Rough hand strength on a 1–4 scale.
          4 = Pair (always wins showdown)
          3 = K, no pair
          2 = Q, no pair
          1 = J, no pair (always loses showdown)
        """
        if self.has_pair:
            return 4
        return self.private_card_rank

    @property
    def is_strong(self) -> bool:
        """Pair or K — almost always wins showdown."""
        return self.hand_strength >= 3

    @property
    def is_weak(self) -> bool:
        """J without a pair — almost always loses showdown."""
        return self.hand_strength == 1


def parse_game_state(obs: str) -> "GameState | None":
    """
    Parse a formatted Leduc Poker observation into a ``GameState``.

    Returns ``None`` if the observation cannot be parsed (e.g. empty or
    "Nothing happens" / "Invalid" messages).
    """
    if not obs or "Current State:" not in obs:
        return None

    def _find(pattern, default=None):
        m = re.search(pattern, obs)
        return m.group(1) if m else default

    # Player
    pid_str = _find(r"You are Player (\d+)\.")
    if pid_str is None:
        return None
    player_id = int(pid_str)

    # Private card  ("Your card: K♠")
    private_card = _find(r"Your card:\s*(\S+)")
    if private_card is None:
        return None
    private_card_rank = _CARD_RANK.get(private_card[0], 0)

    # Public card (optional)
    pub_raw = _find(r"Public card:\s*(\S+)")
    public_card      = pub_raw
    public_card_rank = _CARD_RANK.get(pub_raw[0], 0) if pub_raw else None

    # Pair
    has_pair = "Hand: Pair" in obs

    # Round
    round_str = _find(r"Current round:\s*(\d+)/\d+", "1")
    round_ = int(round_str)

    # Chips
    pot      = int(_find(r"Pot size:\s*(\d+)", "0"))
    our_chips = int(_find(r"Your chips:\s*(\d+)", "100"))
    opp_chips = int(_find(r"Opponent chips:\s*(\d+)", "100"))

    # Betting histories
    def _parse_betting(label: str) -> list[str]:
        m = re.search(rf"{label} betting:\s*(.+)", obs)
        if not m:
            return []
        return [a.strip() for a in m.group(1).split(",")]

    r1_betting = _parse_betting("Round 1")
    r2_betting = _parse_betting("Round 2")

    # Legal actions
    legal_actions: dict[int, str] = {}
    for m in re.finditer(r"^(\d+)\s*->\s*(.+)$", obs, re.MULTILINE):
        legal_actions[int(m.group(1))] = m.group(2).strip()

    return GameState(
        player_id=player_id,
        private_card=private_card,
        private_card_rank=private_card_rank,
        public_card=public_card,
        public_card_rank=public_card_rank,
        has_pair=has_pair,
        round=round_,
        pot=pot,
        our_chips=our_chips,
        opp_chips=opp_chips,
        r1_betting=r1_betting,
        r2_betting=r2_betting,
        legal_actions=legal_actions,
    )


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Shaped reward calculator for Leduc Poker training."""

    SIGNALS = {
        "fold_pair":         -2.0,
        "fold_k":            -1.5,
        "fold_kq_r1_raise":  -1.5,
        "fold_q_pubk_raise": -0.5,
        "fold_j_r1_raise":   +0.3,
        "fold_j_r2_raise":   +0.2,
        "fold_q_pubj_raise": +0.2,
        "raise_pair_r2":     +0.3,
        "raise_k_r2":        +0.2,
        "call_kq_r1_raise":  +0.2,
    }

    def __init__(self, terminal_weight: float = 1.0, gamma: float = 0.9):
        self.terminal_weight = terminal_weight
        self.gamma = gamma

    def calculate_step_reward(
        self,
        gs: "GameState | None",
        action_str: str,
        env_reward: float,
        components: "dict | None" = None,
    ) -> float:
        reward = 0.0
        signal_key: "str | None" = None

        if gs is not None:
            pub = gs.public_card_rank or 0

            if action_str == "Fold":
                if gs.has_pair:
                    signal_key = "fold_pair"
                elif gs.private_card_rank == 3:
                    signal_key = "fold_k"
                elif gs.round == 1 and gs.opp_last_action == "Raise":
                    signal_key = "fold_kq_r1_raise" if gs.private_card_rank >= 2 else "fold_j_r1_raise"
                elif gs.round == 2 and gs.opp_last_action == "Raise":
                    if gs.private_card_rank == 1:
                        signal_key = "fold_j_r2_raise"
                    elif gs.private_card_rank == 2 and pub == 1:
                        signal_key = "fold_q_pubj_raise"
                    elif gs.private_card_rank == 2 and pub == 3:
                        signal_key = "fold_q_pubk_raise"

            elif action_str == "Raise":
                if gs.round == 2 and gs.has_pair:
                    signal_key = "raise_pair_r2"
                elif gs.round == 2 and gs.private_card_rank == 3 and not gs.has_pair:
                    signal_key = "raise_k_r2"

            elif action_str in ("Call", "Check"):
                if gs.round == 1 and gs.opp_last_action == "Raise" and gs.private_card_rank >= 2:
                    signal_key = "call_kq_r1_raise"

        if signal_key is not None:
            reward += self.SIGNALS[signal_key]

        terminal_part = env_reward * self.terminal_weight if env_reward != 0.0 else 0.0
        reward += terminal_part

        if components is not None:
            if signal_key is not None:
                components[signal_key] = components.get(signal_key, 0.0) + self.SIGNALS[signal_key]
            components["terminal"] = components.get("terminal", 0.0) + terminal_part

        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _current_mcts_sims(curriculum: CurriculumScheduler) -> int:
    """
    Progressive MCTS sim ramp, derived from the shared scheduler's existing
    turn progression — no subclassing needed.  Easy→target as the agent
    advances through turn stages.
    """
    turn_range = max(curriculum.final_max_turn - curriculum.initial_max_turn, 1)
    progress = (curriculum.get_max_turn() - curriculum.initial_max_turn) / turn_range
    progress = max(0.0, min(progress, 1.0))
    return int(10 + progress * (50 - 10))


def _ensure_initialized(trainer) -> None:
    """Set up server pool and curriculum once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={_MAX_TURNS}, rollouts_per_stage={trainer.args.rollouts_per_stage}"
    )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
    )


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _format_observation(raw: str) -> str:
    """
    Reformat the server observation to match the eval framework format.

    Server returns:
        # Game Rules\\n...\\n# Current Game State\\nGame: leduc_poker\\n
        You are Player X.\\n\\nCurrent State:\\n...\\nLegal Actions:\\n  N -> ...\\n
        Your choice (action ID only):

    Eval sends:
        Current State:\\n...\\nYour turn to act\\n\\nYou are Player X.\\n
        Legal Actions:\\nN -> ...\\n\\nYour choice (ID only):
    """
    # Extract player number
    player_match = re.search(r"You are Player (\d+)\.", raw)
    player_line  = f"You are Player {player_match.group(1)}." if player_match else ""

    # Find start of "Current State:"
    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw  # unexpected format, pass through

    body = raw[state_start:]

    # Find "Legal Actions:" to split state from actions
    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body

    state_block   = body[:legal_start].rstrip()   # ends with "Your turn to act" (or similar)
    actions_block = body[legal_start:]

    # Remove leading spaces from action lines ("  N -> X" → "N -> X")
    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)

    # Normalise the choice prompt to match eval
    actions_block = actions_block.replace(
        "Your choice (action ID only):", "Your choice (ID only):"
    )

    # Assemble in eval order: state, blank, player, actions
    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n\n".join(parts)


def _pot_odds_line(gs: "GameState | None") -> str:
    """One-line pot-odds cue derived from the parsed game state.

    Pot odds = chips_to_call / (pot + chips_to_call): the minimum win rate at which
    calling is +EV. Returns empty when there's nothing to call.
    """
    if gs is None:
        return ""
    chips_to_call = max(gs.opp_contributed - gs.our_contributed, 0)
    if chips_to_call == 0:
        return f"[Pot odds] Pot:{gs.pot}  To call:0  No bet to face."
    pot_after_call = gs.pot + chips_to_call
    threshold = chips_to_call / pot_after_call if pot_after_call > 0 else 0.0
    return (
        f"[Pot odds] Pot:{gs.pot}  To call:{chips_to_call}  "
        f"Win-rate to break-even: {threshold:.0%} "
        f"(call profitable if your est. win rate \u2265 {threshold:.0%})"
    )


_EOS_SUFFIXES = ("</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")


def _parse_action(completion_text: str) -> str:
    """Robust action extractor: strips common EOS markers + answer prefixes,
    then pulls the first non-negative integer. Returns "" on parse failure."""
    cleaned = completion_text.strip()
    for suffix in _EOS_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip()
    for marker in ("Action:", "action:", "ACTION:", "Answer:", "answer:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
            break
    match = re.search(r"\b\d+\b", cleaned)
    return match.group(0) if match else ""


def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt: bool,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_hint_prob: float,
    current_mcts_sims: int,
) -> tuple[int, "dict | None"]:
    """
    Run one Leduc Poker episode.

    Reward = final game return (e.g. +1.0 win / -1.0 loss), with a small
    penalty applied per invalid action.

    When ``use_full_prompt=True``, accumulates token IDs across all turns with
    action masking (mask=1 for LLM completions, mask=0 for environment tokens).
    When ``use_full_prompt=False``, only the final turn's token IDs are kept.
    """
    game_id = int(prompt)
    server_idx = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt state (overwritten each turn)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done          = False
    final_reward  = 0.0
    episode_reward = 0.0
    turn_number   = 0
    invalid_count = 0
    use_hints     = random.random() < current_hint_prob
    game_state_history: list[GameState] = []
    calculator    = RewardCalculator()
    rewards:        list[float] = []
    components: dict[str, float] = {}
    invalid_penalty_sum: float   = 0.0

    # Reset environment
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": current_mcts_sims,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id  = result_block.get("episode_id", "")
        observation = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
            pot_line = _pot_odds_line(gs)
            if pot_line:
                observation = f"{observation}\n\n{pot_line}"
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": observation},
    ]

    while not done and turn_number < _MAX_TURNS:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (full-prompt mode) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE tokenisers are not context-free: re-tokenising the full
                    # conversation can shift earlier token IDs. Skip delta mask.
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask for this turn."
                    )
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- Parse and send action ---
        action_to_send = _parse_action(completion_text)
        prev_gs = game_state_history[-1] if game_state_history else None

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            step_reward = step_block.get("reward", 0)
            done        = step_block.get("done", False)
            if not done:
                gs = parse_game_state(observation)
                if gs is not None:
                    game_state_history.append(gs)
                    pot_line = _pot_odds_line(gs)
                    if pot_line:
                        observation = f"{observation}\n\n{pot_line}"
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation = ""
            step_reward = 0
            done        = False
            invalid_count += 1
            episode_reward += _INVALID_PENALTY
            invalid_penalty_sum += _INVALID_PENALTY

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count += 1
            episode_reward += _INVALID_PENALTY
            invalid_penalty_sum += _INVALID_PENALTY

        if done:
            final_reward = step_reward

        try:
            action_str = prev_gs.legal_actions.get(int(action_to_send.strip()), "") if prev_gs else ""
        except (ValueError, AttributeError):
            action_str = ""
        step_shaped = calculator.calculate_step_reward(
            prev_gs, action_str, step_reward if done else 0.0, components=components,
        )
        rewards.append(step_shaped)

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    train_reward = calculator.calculate_discounted_return(rewards) + episode_reward
    print(
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{:<2} | EnvR:{:>6.2f} | TrainR:{:>6.2f} | Inv:{:<2}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints), final_reward, train_reward, invalid_count,
        )
    )

    components["invalid_penalty"] = invalid_penalty_sum

    # --- Build result ---
    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
            "invalid_count":  invalid_count,
            "components":     components,
        }
    else:
        return index, {
            "prompt_ids":     prompt_ids,
            "completion_ids": completion_ids,
            "logprobs":       logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
            "invalid_count":  invalid_count,
            "components":     components,
        }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    """Common dispatch + aggregation logic for both rollout variants."""
    _ensure_initialized(trainer)

    curriculum: CurriculumScheduler = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    current_mcts_sims = _current_mcts_sims(curriculum)
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, "
        f"mcts_sims={current_mcts_sims}"
    )

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_hint_prob=current_hint_prob,
        current_mcts_sims=current_mcts_sims,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "invalid_count": 0, "components": {}}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "invalid_count": 0, "components": {}}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    n = len(list_results)
    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    wins       = sum(1 for r in list_results if r.get("final_score", 0) > 0)
    losses     = sum(1 for r in list_results if r.get("final_score", 0) < 0)
    avg_return = sum(r["reward"] for r in list_results) / n if n else 0
    win_rate   = (wins / finished) if finished else 0.0
    avg_invalid = sum(r.get("invalid_count", 0) for r in list_results) / n if n else 0
    print(
        f"[BATCH] Finished:{finished}/{n} W:{wins} L:{losses} "
        f"WinRate:{win_rate:.2%} AvgReturn:{avg_return:.2f} AvgInv:{avg_invalid:.2f}"
    )

    component_keys = list(RewardCalculator.SIGNALS.keys()) + ["terminal", "invalid_penalty"]
    if n:
        avgs = {
            k: sum(r.get("components", {}).get(k, 0.0) for r in list_results) / n
            for k in component_keys
        }
        comp_str = " ".join(f"{k}:{v:+.3f}" for k, v in avgs.items() if abs(v) > 1e-6)
        print(f"[SHAPING] {comp_str}" if comp_str else "[SHAPING] (no signal)")

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
        "terminal_raw":   [float(r["final_score"])                 for r in list_results],
        "shaping_sum":    [float(r["reward"] - r["final_score"])   for r in list_results],
        "invalid_count":  [int(r.get("invalid_count", 0))          for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
