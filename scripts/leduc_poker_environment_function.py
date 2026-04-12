import os
import re
import random
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from trl.experimental.openenv import generate_rollout_completions

GAME_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

SELECTED_GAME = "leduc_poker"
TIMEOUT = 2400

# Action IDs for Leduc Poker
ACTION_FOLD  = "0"
ACTION_CALL  = "1"
ACTION_RAISE = "2"

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 50,
    "mcts_num_rollouts": 1,
}

# Curriculum: Leduc Poker max game length = 8 turns (2 rounds × up to 4 bets each)
CURRICULUM_INITIAL_TURN = 2       # start simple: one bet/response round
CURRICULUM_FINAL_TURN = 8         # full game length
CURRICULUM_ROLLOUTS_PER_STAGE = 512   # 6 stages × 512 = 3072 rollouts to reach max
CURRICULUM_WARMUP_ROLLOUTS = 128  # short warmup before progression starts

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

class CurriculumScheduler:
    """Progressive turn-limit curriculum."""

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 50,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        self.total_rollouts += num_rollouts

def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()

def parse_action(completion_text: str) -> str:
    """Strip reasoning tags and ReAct 'Action:' prefix, return raw action string."""
    action = remove_reasoning_tags(completion_text)
    if action.endswith("</s>"):
        action = action[:-4]
    if "Action:" in action:
        action = action.split("Action:")[-1]
    return action.strip()    

def extract_and_format_observation(obs_text: str) -> str:
    if "Invalid action:" in obs_text and "Legal Actions:" in obs_text:
        return obs_text
    state_match = re.search(r"Current State:\n(.*)", obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)
    player_match = re.search(r"You are Player (\d+)", obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    body, _, legal = state_text.partition("Legal Actions:")
    return body + f"You are Player {player_id}.\nLegal Actions:" + legal


def parse_poker_state(obs: str) -> dict:
    """Extract key fields from observation for reward shaping."""
    state = {"private_card": None, "public_card": None, "pot": 0, "round": 1, "my_chips": 100, "has_pair": False}
    m = re.search(r"Your card:\s*([JQKA][♠♥♦♣])", obs)
    if m: state["private_card"] = m.group(1)
    m = re.search(r"Public card:\s*([JQKA][♠♥♦♣])", obs)
    if m: state["public_card"] = m.group(1)
    m = re.search(r"Pot size:\s*(\d+)", obs)
    if m: state["pot"] = int(m.group(1))
    m = re.search(r"Current round:\s*(\d+)", obs)
    if m: state["round"] = int(m.group(1))
    m = re.search(r"Your chips:\s*(\d+)", obs)
    if m: state["my_chips"] = int(m.group(1))
    if "Hand: Pair" in obs: state["has_pair"] = True
    # Check pair by comparing ranks of private and public
    if state["private_card"] and state["public_card"]:
        priv_rank = state["private_card"][0]
        pub_rank  = state["public_card"][0]
        if priv_rank == pub_rank:
            state["has_pair"] = True
    return state


# Card rank order J < Q < K < A
_RANK_STRENGTH = {"J": 1, "Q": 2, "K": 3, "A": 4}


class RewardCalculator:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        # Tuned for MCTS(50,1) — 2× stronger opponent than previous (25,1).
        # Reduced shaping magnitudes so terminal outcome dominates (avoids reward hacking).
        self.pair_bonus      =  5.0   # was 8.0 — less dominant, agent learns mixed strategy
        self.high_card_bonus =  1.5   # was 2.0
        self.fold_penalty    = -3.0   # was -5.0 — correct folds vs strong opponent less penalised

    def calculate_step_reward(self, prev_state, curr_state, action, is_invalid, is_terminal, final_env_reward=0.0):
        if is_terminal:
            return max(min(final_env_reward * 30.0, 30.0), -30.0)  # was ×50; reduced for GRPO stability
        if is_invalid:
            return -5.0
        reward = 0.0
        if curr_state.get("has_pair"):
            reward += self.pair_bonus
        elif curr_state.get("private_card"):
            reward += self.high_card_bonus * _RANK_STRENGTH.get(curr_state["private_card"][0], 1)
        # Penalize fold without pair on round 1 (overly passive)
        if action == ACTION_FOLD and curr_state.get("round", 1) == 1 and not curr_state.get("has_pair"):
            reward += self.fold_penalty * 0.5
        # Pot-growth signal
        if prev_state and curr_state.get("pot", 0) > prev_state.get("pot", 0):
            reward += 0.3  # was 0.5 — scaled down with other shaping
        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        G = 0.0
        for r in reversed(rewards): G = r + self.gamma * G
        return G




_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n# Game Rules\nLEDUC POKER RULES:\n"
    "Deck: 6 cards (J♠ J♥ Q♠ Q♥ K♠ K♥). Each player starts with 100 chips, pays 1 ante.\n"
    "Round 1: Receive 1 private card. Fold(0), Call(1), Raise(2). Max 2 raises.\n"
    "Round 2: Public card revealed. Same actions, Raise adds 4 chips.\n"
    "Hand Ranking: Pair (private matches public) > High card (K>Q>J).\n\n"
    "# Output Format\nRespond with ONLY the action ID (a single number).\n"
    "Actions: 0=Fold, 1=Call/Check, 2=Raise\n"
    'Example: For action "1 -> call": respond "1"'
)
_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "- With a pair (private matches public) → raise aggressively\n"
    "- With K → call or raise in round 1, see the public card\n"
    "- With J → fold to aggressive raises if no pair\n"
    "- Track opponent betting pattern to detect bluffs"
)


def _ensure_initialized(fn, trainer) -> None:
    if getattr(fn, "initialized", False): return
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    server_urls = [u.strip() for u in os.environ.get("ENVIRONMENT_SERVER_URLS", "").split(",") if u.strip()]
    if not server_urls: raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")
    env_pool = []
    init_payload = {"task_id": GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0], "seed": 42, **MCTS_CONFIG}
    for idx, base_url in enumerate(server_urls):
        res = requests.post(f"{base_url}/reset", json=init_payload, timeout=300)
        res.raise_for_status(); env_pool.append({"base_url": base_url}); print(f"[INIT] Server {idx} ready")
    fn.rank = rank; fn.env_pool = env_pool; fn.num_servers = len(env_pool)
    fn.thread_pool = ThreadPoolExecutor(max_workers=len(env_pool)); fn.generation_semaphore = Semaphore(1)
    fn.curriculum = CurriculumScheduler(
        initial_max_turn=CURRICULUM_INITIAL_TURN,
        final_max_turn=CURRICULUM_FINAL_TURN,
        rollouts_per_stage=CURRICULUM_ROLLOUTS_PER_STAGE,
        warmup_rollouts=CURRICULUM_WARMUP_ROLLOUTS,
    )
    fn.initialized = True
    print(f"[CURRICULUM] Initialized: turns {CURRICULUM_INITIAL_TURN}→{CURRICULUM_FINAL_TURN}, mcts_sims={MCTS_CONFIG['mcts_max_simulations']}")


def rollout_last_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30):
    _ensure_initialized(rollout_last_prompt_and_completion_parallelized_curriculum, trainer)
    fn = rollout_last_prompt_and_completion_parallelized_curriculum
    tokenizer = trainer.processing_class
    current_max_turn = fn.curriculum.get_max_turn(); current_hint_prob = fn.curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {fn.curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single(index, prompt):
        game_id = int(prompt); env_endpoint = fn.env_pool[(index + fn.rank) % fn.num_servers]["base_url"]
        inv_cnt = srv_cnt = 0; done = False; final_reward = 0.0; turn_number = 0
        rewards = []; prev_state = None; prompt_ids = []; completion_ids = []; logprobs = []
        use_hints = random.random() < current_hint_prob; calculator = RewardCalculator()

        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}
        try:
            res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            res.raise_for_status(); rb = res.json()["result"]
            episode_id = rb.get("episode_id", ""); fmt_obs = extract_and_format_observation(rb.get("observation", ""))
        except Exception as exc: print(f"[LAST] Reset failed (game {game_id}): {exc}"); return index, None

        system_prompt = _SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": fmt_obs}]
        curr_state = parse_poker_state(fmt_obs)

        while not done and turn_number < current_max_turn:
            with fn.generation_semaphore:
                out = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = out.get("prompt_ids", []); completion_ids = out.get("completion_ids", []); logprobs = out.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": completion_text}); action_to_send = parse_action(completion_text)
            try:
                sr = requests.post(f"{env_endpoint}/step", json={"action": action_to_send, "episode_id": episode_id}, timeout=TIMEOUT)
                sr.raise_for_status(); sb = sr.json()["result"]
                fmt_obs = extract_and_format_observation(sb.get("observation", ""))
                step_reward = sb.get("reward", 0); done = sb.get("done", False)
            except Exception as exc: print(f"[LAST] Step failed: {exc}"); step_reward = -0.01; done = False; srv_cnt += 1
            is_invalid = "Nothing happens" in fmt_obs or "Invalid" in fmt_obs
            if is_invalid: inv_cnt += 1
            if done: final_reward = step_reward
            else: messages.append({"role": "user", "content": fmt_obs})
            next_state = parse_poker_state(fmt_obs)
            imm = calculator.calculate_step_reward(curr_state, next_state, action_to_send, is_invalid, done, final_reward)
            prev_state = curr_state; curr_state = next_state; rewards.append(imm); turn_number += 1

        train_reward = rewards[-1] if rewards else 0.0
        print(f"[LAST] id={game_id} done={int(done)} T={turn_number} reward={train_reward:.2f} envR={final_reward:.1f} inv={inv_cnt}")
        return index, {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs, "reward": train_reward, "final_score": final_reward}

    results = [None] * len(prompts)
    for f in as_completed([fn.thread_pool.submit(run_single, i, p) for i, p in enumerate(prompts)]):
        idx, res = f.result()
        results[idx] = res if res is not None else {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    fn.curriculum.step(len(prompts)); valid = [r for r in results if r is not None]
    avg_ret = sum(r["reward"] for r in valid) / len(valid) if valid else 0
    print(f"[LAST-BATCH] AvgReturn: {avg_ret:.2f}")
    return {"prompt_ids": [r["prompt_ids"] for r in results], "completion_ids": [r["completion_ids"] for r in results], "logprobs": [r["logprobs"] for r in results], "env_rewards": [r["reward"] for r in results]}


def rollout_full_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30):
    MAX_EPISODE_TOKENS = 16_384; MAX_PROMPT_LEN = 5_000
    _ensure_initialized(rollout_full_prompt_and_completion_parallelized_curriculum, trainer)
    fn = rollout_full_prompt_and_completion_parallelized_curriculum; tokenizer = trainer.processing_class
    current_max_turn = fn.curriculum.get_max_turn(); current_hint_prob = fn.curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {fn.curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single(index, prompt):
        game_id = int(prompt); env_endpoint = fn.env_pool[(index + fn.rank) % fn.num_servers]["base_url"]
        ep_prompt_ids = []; ep_comp_ids = []; ep_logprobs = []; ep_mask = []; prev_full_ids = None
        inv_cnt = srv_cnt = 0; done = False; final_reward = 0.0; turn_number = 0
        rewards = []; prev_state = None; calculator = RewardCalculator()
        use_hints = random.random() < current_hint_prob

        payload = {"task_id": game_id, "seed": game_id, "opponent": "mcts", "mcts_max_simulations": 50, "mcts_num_rollouts": 1}
        try:
            res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            res.raise_for_status(); rb = res.json()["result"]
            episode_id = rb.get("episode_id", ""); fmt_obs = extract_and_format_observation(rb.get("observation", ""))
        except Exception as exc: print(f"[FULL] Reset failed (game {game_id}): {exc}"); return index, None

        system_prompt = _SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": fmt_obs}]
        curr_state = parse_poker_state(fmt_obs)

        while not done and turn_number < current_max_turn:
            with fn.generation_semaphore:
                out = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = out.get("prompt_ids", []); completion_ids = out.get("completion_ids", []); logprobs = out.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            if len(prompt_ids) > MAX_PROMPT_LEN: done = True; break
            if turn_number == 0: ep_prompt_ids = prompt_ids; prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None: prev_full_ids = prompt_ids.copy()
                elif prompt_ids[:len(prev_full_ids)] != prev_full_ids: prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta: ep_comp_ids.extend(delta); ep_logprobs.extend([0.0]*len(delta)); ep_mask.extend([0]*len(delta))
                    prev_full_ids = prompt_ids.copy()
            if completion_ids:
                ep_comp_ids.extend(completion_ids); ep_logprobs.extend(logprobs); ep_mask.extend([1]*len(completion_ids))
                if prev_full_ids is not None: prev_full_ids = prev_full_ids + completion_ids
            messages.append({"role": "assistant", "content": completion_text}); action_to_send = parse_action(completion_text)
            try:
                sr = requests.post(f"{env_endpoint}/step", json={"action": action_to_send, "episode_id": episode_id}, timeout=TIMEOUT)
                sr.raise_for_status(); sb = sr.json()["result"]
                fmt_obs = extract_and_format_observation(sb.get("observation", ""))
                step_reward = sb.get("reward", 0); done = sb.get("done", False)
            except Exception as exc: print(f"[FULL] Step failed: {exc}"); step_reward = 0.0; done = False; srv_cnt += 1
            is_invalid = "Nothing happens" in fmt_obs or "Invalid" in fmt_obs
            if is_invalid: inv_cnt += 1
            messages.append({"role": "user", "content": fmt_obs})
            if done: final_reward = step_reward
            next_state = parse_poker_state(fmt_obs)
            imm = calculator.calculate_step_reward(curr_state, next_state, action_to_send, is_invalid, done, final_reward)
            prev_state = curr_state; curr_state = next_state; rewards.append(imm); turn_number += 1

        if len(ep_comp_ids) > MAX_EPISODE_TOKENS:
            ep_comp_ids = ep_comp_ids[:MAX_EPISODE_TOKENS]; ep_logprobs = ep_logprobs[:MAX_EPISODE_TOKENS]; ep_mask = ep_mask[:MAX_EPISODE_TOKENS]
        disc_ret = calculator.calculate_discounted_return(rewards)
        print(f"[FULL] id={game_id} done={int(done)} T={turn_number} ret={disc_ret:.2f} envR={final_reward:.1f} inv={inv_cnt}")
        return index, {"prompt_ids": ep_prompt_ids, "completion_ids": ep_comp_ids, "action_mask": ep_mask, "logprobs": ep_logprobs, "reward": disc_ret, "final_score": final_reward}

    results = [None] * len(prompts)
    for f in as_completed([fn.thread_pool.submit(run_single, i, p) for i, p in enumerate(prompts)]):
        idx, res = f.result()
        results[idx] = res if res is not None else {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    fn.curriculum.step(len(prompts)); valid = [r for r in results if r is not None]
    avg_ret = sum(r["reward"] for r in valid) / len(valid) if valid else 0; print(f"[FULL-BATCH] AvgReturn: {avg_ret:.2f}")
    return {"prompt_ids": [r["prompt_ids"] for r in results], "completion_ids": [r["completion_ids"] for r in results], "action_mask": [r["action_mask"] for r in results], "logprobs": [r["logprobs"] for r in results], "env_rewards": [r["reward"] for r in results]}


def rollout_reward_func(completions, **kwargs) -> list[float]:
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
