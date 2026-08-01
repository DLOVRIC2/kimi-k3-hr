"""Apply Kimi-K3's chat template without transformers.

K3 is instruction-tuned. Feeding it raw text makes it behave like a base model --
it continues your prompt instead of answering it. That is not a capability
failure, and mistaking one for the other is how you end up "discovering" that a
build cannot follow instructions when in fact it was never asked to.

The shipped tokenization_kimi.py owns apply_chat_template but subclasses
transformers' PreTrainedTokenizer and targets transformers 4.56, which does not
import under 5.x. encoding_k3.py -- where the actual template lives -- has no
transformers dependency, so we import that directly and do the segment->ids step
ourselves. It is six lines, and it is exactly what _encode_chat_segments does.

The template is XTML-tagged, not Llama-3 headers:

    <|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
    <|open|>message role="assistant"<|sep|><|open|>response<|sep|>

thinking=True instead opens a `think` channel and prepends a thinking-effort
system message. Default it OFF here: at ~5 tok/s a reasoning channel consumes
the whole generation budget before any answer appears.
"""

from __future__ import annotations

import functools
import json
import pathlib
import sys
from typing import Any

BOS = 163584
EOS = 163585
END_OF_MSG = 163586
OPEN = 163587
CLOSE = 163588
SEP = 163589

# CLOSE ends the assistant's response channel, so it is a stop as much as
# END_OF_MSG -- without it every generation trails its own closing markup.
STOP_TOKENS = {EOS, END_OF_MSG, CLOSE}


@functools.lru_cache(maxsize=4)
def _load_builder(src_dir: str):
    """Import build_chat_segments from the model directory itself."""
    src = str(pathlib.Path(src_dir).expanduser())
    if src not in sys.path:
        sys.path.insert(0, src)
    from encoding_k3 import build_chat_segments  # noqa: PLC0415
    return build_chat_segments


@functools.lru_cache(maxsize=4)
def _special_map(src_dir: str) -> dict[str, int]:
    """Real name -> id for the control tokens.

    This is the whole reason this module exists rather than calling
    enc.encode(text, allowed_special="all"). The toolchain's build_tokenizer
    registers the high vocab range as generic `<|reserved_special_token_N|>`
    placeholders, so tiktoken has never heard of "<|open|>" and happily encodes
    it as five ORDINARY tokens. The model then sees the chat markup as literal
    text, echoes the prompt back, and degenerates -- which looks exactly like a
    broken model rather than a broken tokenizer.

    tokenizer_config.json carries the true mapping, so read it from there.
    """
    cfg = pathlib.Path(src_dir).expanduser() / "tokenizer_config.json"
    atd = json.loads(cfg.read_text()).get("added_tokens_decoder", {})
    return {v["content"]: int(k) for k, v in atd.items()}


def render(
    src_dir: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    thinking: bool = False,
    thinking_effort: str | None = None,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> str:
    """Return the templated prompt as text -- useful for eyeballing it."""
    build = _load_builder(src_dir)
    if thinking and thinking_effort:
        kwargs["thinking_effort"] = thinking_effort
    segs = build(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        thinking=thinking,
        **kwargs,
    )
    return "".join(s.text for s in segs)


def encode(
    enc,
    src_dir: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    thinking: bool = False,
    thinking_effort: str | None = None,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> list[int]:
    """Template the messages and return token ids.

    Control segments must be encoded with special tokens ALLOWED, ordinary text
    with them disallowed -- otherwise a user who types "<|sep|>" would inject
    control tokens into the conversation.
    """
    build = _load_builder(src_dir)
    if thinking and thinking_effort:
        kwargs["thinking_effort"] = thinking_effort
    segs = build(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        thinking=thinking,
        **kwargs,
    )

    specials = _special_map(src_dir)
    ids: list[int] = []
    for seg in segs:
        if seg.allow_special:
            tid = specials.get(seg.text)
            if tid is None:
                raise ValueError(
                    f"control segment {seg.text!r} is not in tokenizer_config.json's "
                    "added_tokens_decoder. Encoding it as text would silently corrupt "
                    "the conversation, so refusing."
                )
            ids.append(tid)
        else:
            ids.extend(enc.encode_ordinary(seg.text))
    return ids


CONTROL_IDS = {BOS, EOS, END_OF_MSG, OPEN, CLOSE, SEP}


def split_channels(ids: list[int]) -> tuple[list[int], list[int]]:
    """Split generated ids into (thinking, response).

    With thinking on, K3 emits its chain of thought and then opens a second
    channel -- `<|open|>response<|sep|>` -- before the actual answer. Two things
    go wrong if that is ignored:

    1. The answer is scored against the THINKING text. A multiple-choice regex
       then matches whatever digit the model reasoned about first, which is
       uncorrelated with its conclusion. Measured 30% on Croatian, i.e. chance.
    2. `enc.decode` renders the control tokens as the generic
       `<|reserved_special_token_N|>` placeholders the toolchain registered, so
       the output looks like '<|reserved_special_token_3|>response...'. Same root
       cause as the encode-side bug, mirrored.

    The response channel is whatever follows the LAST separator; control tokens
    are dropped from it. With thinking off there is no separator, so this is a
    no-op apart from stripping stray control ids.
    """
    if SEP in ids:
        i = len(ids) - 1 - ids[::-1].index(SEP)
        return ids[:i], [t for t in ids[i + 1:] if t not in CONTROL_IDS]
    return [], [t for t in ids if t not in CONTROL_IDS]


def user(text: str) -> list[dict[str, Any]]:
    """Shorthand for the common single-turn case."""
    return [{"role": "user", "content": text}]
