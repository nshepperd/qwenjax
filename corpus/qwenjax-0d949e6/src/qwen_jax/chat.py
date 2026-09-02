"""The ChatML template Qwen3 uses, as token-level pieces.

Written out rather than going through `apply_chat_template` because cartridge
training needs the *same* suffix token ids in two different contexts -- after
a real system prompt (teacher) and after a cartridge (student) -- and the only
way to guarantee that is to tokenize each piece on its own and concatenate ids.
Special tokens are hard boundaries for the BPE anyway, so this produces the
tokenization the official template would.

The cartridge takes the place of `<|im_start|>system\\n{document}`; what
follows it, `<|im_end|>\\n<|im_start|>user\\n...`, is the `suffix`.
"""
from __future__ import annotations

from dataclasses import dataclass

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_OPEN = f"{IM_START}system\n"


def system_open(text: str) -> str:
    """The opening of a system turn, up to but not including its `<|im_end|>`.

    This is what a cartridge is initialised from: the prefix a real document
    would occupy.
    """
    return SYSTEM_OPEN + text


def suffix(turns: list[tuple[str, str]], *, open_assistant: bool = False) -> str:
    """Everything after the system text: its close, then the conversation.

    `turns` alternate `(role, text)`. With `open_assistant`, ends with an open
    assistant turn for the model to continue -- the generation prompt.
    """
    out = f"{IM_END}\n"
    for role, text in turns:
        out += f"{IM_START}{role}\n{text}{IM_END}\n"
    if open_assistant:
        out += f"{IM_START}assistant\n"
    return out


@dataclass(frozen=True)
class Tokens:
    """Token ids of a prompt split at the system/suffix boundary."""

    system: list[int]
    suffix: list[int]

    @property
    def ids(self) -> list[int]:
        return self.system + self.suffix


def encode(tokenizer, system_text: str, turns: list[tuple[str, str]],
           *, open_assistant: bool = False) -> Tokens:
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    return Tokens(
        system=enc(system_open(system_text)),
        suffix=enc(suffix(turns, open_assistant=open_assistant)),
    )


__all__ = ["IM_END", "IM_START", "SYSTEM_OPEN", "Tokens", "encode", "suffix", "system_open"]
