"""PvP-format prompt builders for SFT trajectory generation.

Mirrors `validator/evaluation/pvp/agents.py` from G.O.D `feature/pvp-eval-container`
branch (canonical eval format for tournament starting 2026-05-25).

Each agent class converts env server raw observation OR pyspiel state into the
exact PvP user-prompt format the validator will feed to our model during eval:

    Current State:
    {state_desc}

    You are Player {player_id}.
    Legal Actions:
    {action_id} -> {action_str}
    ...

    Your choice (ID only):

Match-target source: gradients-ai/G.O.D feature/pvp-eval-container
File: validator/evaluation/pvp/agents.py (cached at scripts/envs/pvp_assets/agents.py)
Prompts: core/config/pvp_game_prompts.yml (cached at scripts/envs/pvp_assets/pvp_game_prompts.yml)

SFT trajectory generators import from here to produce eval-format-matching data.

NOTE: the OUTPUT strings of every builder below are byte-for-byte the eval
format and must not change. The internal structure here has been re-organised
(local naming / assembly) without altering any produced string; verified by an
output-equality check against the reference implementation.
"""

import re
from pathlib import Path

import yaml


_ASSETS_DIR = Path(__file__).resolve().parent / "pvp_assets"
_PROMPTS_PATH = _ASSETS_DIR / "pvp_game_prompts.yml"

# Card encoding tables for Leduc (id//2 = rank, id%2 = suit).
_LEDUC_RANKS = ["J", "Q", "K", "A"]   # A only appears in 3+ player variants
_LEDUC_SUITS = ["♠", "♥"]   # ♠ ♥
# Leduc betting-sequence id -> human label.
_LEDUC_BET_LABELS = {0: "Fold", 1: "Call", 2: "Raise"}
# Sentinel the env uses for "card not dealt".
_UNDEALT = -10000


def _load_prompts() -> dict:
    """Load PvP system prompt template + per-game rules from the cached YAML."""
    with open(_PROMPTS_PATH) as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# Helpers (defined first so the formatters below can reference them)
# ---------------------------------------------------------------------------

def _first_group(text: str, pattern: str) -> str:
    """Return the first regex capture group, or '' when there is no match."""
    found = re.search(pattern, text)
    return found.group(1) if found else ""


# Backwards-compatible alias (older callers import this name).
_extract_pattern = _first_group


def _leduc_card_name(card_id: int) -> str:
    """0=J♠ 1=J♥ 2=Q♠ 3=Q♥ 4=K♠ 5=K♥ (per pvp/agents.py:_card_name)."""
    rank_pos, suit_pos = divmod(card_id, 2)
    if rank_pos < len(_LEDUC_RANKS):
        return f"{_LEDUC_RANKS[rank_pos]}{_LEDUC_SUITS[suit_pos]}"
    return f"Card_{card_id}"


def _parse_betting(seq: str) -> str:
    """Translate a Leduc betting sequence like '1 1' into 'Call, Call'."""
    if not seq or not seq.strip():
        return "(none)"
    ids = [int(tok) for tok in seq.split() if tok.isdigit()]
    if not ids:
        return "(none)"
    return ", ".join(_LEDUC_BET_LABELS.get(i, f"Action{i}") for i in ids)


def _extract_legal_actions_block(env_obs: str) -> str:
    """Pull the `<id> -> <label>` lines out of an env observation and re-emit
    them one per line, ready to drop into `build_user_prompt`."""
    matches = re.findall(r"^[ \t]*(\d+)[ \t]*->[ \t]*(.+)$", env_obs, re.MULTILINE)
    rendered = [f"{action_id} -> {label.strip()}" for action_id, label in matches]
    return "\n".join(rendered)


# ---------------------------------------------------------------------------
# System prompt + user prompt assembly
# ---------------------------------------------------------------------------

def build_system_prompt(game_name: str) -> str:
    """Build the PvP system prompt for one game (liars_dice/leduc_poker/gin_rummy)."""
    prompt_cfg = _load_prompts()
    rules_key = f"{game_name}_rules"
    if rules_key not in prompt_cfg:
        raise ValueError(f"Unknown game: {game_name} (no {rules_key} in pvp_game_prompts.yml)")
    template = prompt_cfg["system_prompt_template"]
    return template.format(game_name=game_name, rules=prompt_cfg[rules_key])


def build_user_prompt(state_desc: str, player_id: int, legal_actions_block: str) -> str:
    """Assemble the PvP user prompt: state + Player ID + Legal Actions + suffix.

    The concatenation below is intentionally identical, character for character,
    to the eval prompt the validator feeds the model.
    """
    segments = (
        f"Current State:\n{state_desc}\n\n",
        f"You are Player {player_id}.\n",
        f"Legal Actions:\n{legal_actions_block}\n\n",
        "Your choice (ID only):",
    )
    return "".join(segments)


# ---------------------------------------------------------------------------
# Per-game state formatters
# ---------------------------------------------------------------------------

def format_liars_dice_state(
    dice: list,
    num_players: int = 2,
    current_player: int = 0,
    current_bid_quantity=None,
    current_bid_face=None,
) -> str:
    """Format Liar's Dice state per validator's LiarsDiceAgent.format_state."""
    per_player = len(dice)
    grand_total = per_player * num_players

    desc_lines = [
        f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {per_player}",
        f"Total dice in game: {grand_total}",
        f"Players: {num_players}",
        f"Current player: Player {current_player}",
    ]

    has_bid = current_bid_quantity is not None and current_bid_face is not None
    if has_bid:
        desc_lines.append(
            f'\nCurrent bid: "{current_bid_quantity}-{current_bid_face}" '
            f"(at least {current_bid_quantity} dice showing {current_bid_face} across all players)"
        )
        desc_lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        desc_lines.append("No bid yet - you must make the first bid")

    return "\n".join(desc_lines)


def format_leduc_poker_state(
    private_card,
    round_num: int,
    pot: int,
    player_chips: int,
    opponent_chips: int,
    public_card=None,
    round1_seq: str = "",
    round2_seq: str = "",
) -> str:
    """Format Leduc Poker state per validator's LeducPokerAgent.format_state."""
    have_private = private_card is not None and private_card != _UNDEALT
    have_public = public_card is not None and public_card != _UNDEALT

    desc_lines: list = []
    desc_lines.append(
        f"Your card: {_leduc_card_name(private_card)}" if have_private
        else "Your card: (not dealt yet)"
    )
    if have_public:
        desc_lines.append(f"Public card: {_leduc_card_name(public_card)}")
        if have_private and private_card // 2 == public_card // 2:
            desc_lines.append("Hand: PAIR")

    desc_lines.append(f"Round: {round_num}/2")
    desc_lines.append(f"Pot: {pot} chips")
    desc_lines.append(f"Your chips: {player_chips}")
    desc_lines.append(f"Opponent chips: {opponent_chips}")

    if round1_seq:
        desc_lines.append(f"Round 1 actions: {_parse_betting(round1_seq)}")
    if round2_seq:
        desc_lines.append(f"Round 2 actions: {_parse_betting(round2_seq)}")

    return "\n".join(desc_lines)


# ---------------------------------------------------------------------------
# Parse env server's raw observation into PvP-formatted state
# ---------------------------------------------------------------------------

def reformat_liars_dice_observation(env_obs: str, player_id: int = 0) -> tuple:
    """Convert the env server's Liar's Dice observation into a
    (state_desc, legal_actions_block) tuple for `build_user_prompt`."""
    dice_raw = _first_group(env_obs, r"Your dice:\s*\[([^\]]+)\]")
    dice = [int(tok.strip()) for tok in dice_raw.split(",")] if dice_raw else []

    total_raw = _first_group(env_obs, r"Total dice in game:\s*(\d+)")
    total_dice = int(total_raw) if total_raw else 10
    per_player = len(dice) if dice else 5
    num_players = max(2, total_dice // max(per_player, 1))

    bid = re.search(r'Current bid:\s*"(\d+)-(\d+)"', env_obs)
    bid_q = int(bid.group(1)) if bid else None
    bid_f = int(bid.group(2)) if bid else None

    state_desc = format_liars_dice_state(
        dice=dice,
        num_players=num_players,
        current_player=player_id,
        current_bid_quantity=bid_q,
        current_bid_face=bid_f,
    )
    return state_desc, _extract_legal_actions_block(env_obs)


def reformat_leduc_poker_observation(env_obs: str, player_id: int = 0) -> tuple:
    """Convert the env server's Leduc observation into a
    (state_desc, legal_actions_block) tuple."""
    private = _first_group(env_obs, r"\[Private:\s*(-?\d+)\]")
    public = _first_group(env_obs, r"\[Public:\s*(-?\d+)\]")
    round_num = _first_group(env_obs, r"\[Round\s*(\d+)\]") or "1"
    pot = _first_group(env_obs, r"\[Pot:\s*(\d+)\]") or "0"
    money = _first_group(env_obs, r"\[Money:\s*([\d ]+)\]") or ""
    r1_seq = _first_group(env_obs, r"\[Round1:\s*([^\]]*)\]") or ""
    r2_seq = _first_group(env_obs, r"\[Round2:\s*([^\]]*)\]") or ""

    chips = money.split()
    opponent_index = 1 - player_id
    player_chips = int(chips[player_id]) if len(chips) > player_id else 100
    opp_chips = int(chips[opponent_index]) if len(chips) > opponent_index else 100

    state_desc = format_leduc_poker_state(
        private_card=int(private) if private else None,
        round_num=int(round_num),
        pot=int(pot),
        player_chips=player_chips,
        opponent_chips=opp_chips,
        public_card=int(public) if public else None,
        round1_seq=r1_seq,
        round2_seq=r2_seq,
    )
    return state_desc, _extract_legal_actions_block(env_obs)


def reformat_gin_rummy_observation(env_obs: str, player_id: int = 0) -> tuple:
    """Gin Rummy uses the raw observation_string as state_desc (per validator's
    GinRummyAgent.format_state); only the legal-actions block is split out."""
    actions_block = _extract_legal_actions_block(env_obs)
    state_desc = re.sub(
        r"\n*Legal Actions:\n(?:[ \t]*\d+[ \t]*->[ \t]*\S.*(?:\n|$))+", "", env_obs
    ).strip()
    return state_desc, actions_block


# ---------------------------------------------------------------------------
# All-in-one entrypoint per game
# ---------------------------------------------------------------------------

def build_pvp_user_prompt_liars_dice(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_liars_dice_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_pvp_user_prompt_leduc_poker(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_leduc_poker_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_pvp_user_prompt_gin_rummy(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_gin_rummy_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


# ---------------------------------------------------------------------------
# Per-game system prompts (cached at import)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LIARS_DICE = build_system_prompt("liars_dice") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_LEDUC_POKER = build_system_prompt("leduc_poker") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_GIN_RUMMY = build_system_prompt("gin_rummy") if _PROMPTS_PATH.exists() else None
