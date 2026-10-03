# SPDX-License-Identifier: Apache-2.0
"""Thinker request builder: length_penalty is checked before the request is queued."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request


def build(params: dict) -> None:
    build_sglang_thinker_request(
        MiniCPMOPipelineState(prompt={"input_ids": torch.tensor([1, 2, 3])}),
        params=params,
        tokenizer=None,
        vocab_size=32000,
    )


def thinker_params(length_penalty: object) -> dict:
    return {"stage_params": {"thinker": {"length_penalty": length_penalty}}}


@pytest.mark.parametrize(
    "params",
    [{}, {"stage_params": {"thinker": None}}, thinker_params(1.0), thinker_params(1.3)],
)
def test_missing_or_positive_length_penalty_builds(params: dict) -> None:
    build(params)


@pytest.mark.parametrize(
    ("length_penalty", "error"),
    [
        (None, TypeError),
        ("1.3", TypeError),
        (0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_invalid_length_penalty_fails_at_build(
    length_penalty: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        build(thinker_params(length_penalty))
