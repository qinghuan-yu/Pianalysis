"""Finite-state generation constraints for the TSD V2 event grammar."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from token_protocol_v1 import TOKEN
from tokenizer_v2 import V2_FIELDS, V2_RANGES


class DurationGrammar(LogitsProcessor):
    def __init__(self, prompt_length: int, budget: int, horizon_ticks: int | None = None):
        self.prompt_length = prompt_length
        self.budget = budget
        self.horizon_ticks = horizon_ticks
        self.forced_eos_reason = None
        if horizon_ticks is not None and horizon_ticks <= 0:
            raise ValueError("Generation horizon must be positive")

    def _state(self, tokens):
        current = 0
        i = 0
        while i < len(tokens):
            op = tokens[i]
            if op == TOKEN["EOS"]:
                return "ended", current
            if op not in (TOKEN["TIME"], TOKEN["NOTE_ON_ACCOMP"]):
                raise ValueError(f"Invalid constrained event prefix: {op}")
            fields = V2_FIELDS[op]
            values = []
            for offset, field in enumerate(fields, 1):
                if i + offset >= len(tokens):
                    return field, current
                base, count, minimum = V2_RANGES[field]
                token = tokens[i + offset]
                if not base <= token < base + count:
                    raise ValueError(f"Invalid constrained {field} prefix: {token}")
                values.append(token - base + minimum)
            if op == TOKEN["TIME"]:
                current += values[0]
            i += len(fields) + 1
        return "boundary", current

    def __call__(self, input_ids, scores):
        for row, prefix in enumerate(input_ids[:, self.prompt_length:].tolist()):
            state, current = self._state(prefix)
            allowed = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            remaining = self.budget - len(prefix)
            time_left = None if self.horizon_ticks is None else self.horizon_ticks - current
            if state in ("boundary", "ended"):
                allowed[TOKEN["EOS"]] = True
                if state != "ended" and (time_left is None or time_left > 0):
                    # 开始一个事件前，给它的所有字段和最终 EOS 留足预算。
                    if remaining >= 5:
                        allowed[TOKEN["NOTE_ON_ACCOMP"]] = True
                    if remaining >= 3:
                        allowed[TOKEN["TIME"]] = True
                if int(allowed.sum()) == 1:
                    self.forced_eos_reason = "horizon" if time_left is not None and time_left <= 0 else "budget"
            else:
                base, count, minimum = V2_RANGES[state]
                if state in ("time", "duration") and time_left is not None:
                    count = min(count, time_left)
                if count <= 0:
                    raise ValueError("No valid event value remains within the generation horizon")
                allowed[base:base + count] = True
            scores[row].masked_fill_(~allowed, -float("inf"))
            if not torch.isfinite(scores[row]).any():
                raise ValueError("No finite logits remain after duration grammar constraints")
        return scores
