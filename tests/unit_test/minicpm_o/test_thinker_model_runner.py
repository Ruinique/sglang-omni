# SPDX-License-Identifier: Apache-2.0
"""Thinker length penalty: parse stage_params and scale EOS logits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minicpm_o.request_builders import resolve_thinker_length_penalty
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.proto import OmniRequest, StagePayload

EOS_TOKEN_IDS = [1, 3]


def bare_runner() -> MiniCPMOThinkerModelRunner:
    runner = object.__new__(MiniCPMOThinkerModelRunner)
    runner.eos_token_ids = EOS_TOKEN_IDS
    return runner


def sampling_params() -> SimpleNamespace:
    return SimpleNamespace(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_new_tokens=0,
        sampling_seed=None,
        logit_bias=None,
        custom_params=None,
    )


def text_payload(length_penalty: float) -> StagePayload:
    return StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=None,
            params={"stage_params": {"thinker": {"length_penalty": length_penalty}}},
            metadata={"output_modalities": ["text"]},
        ),
        data=None,
    )


def schedule_request(length_penalty: float) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=sampling_params(),
        omni_data=SimpleNamespace(
            return_logprob=False,
            stage_payload=text_payload(length_penalty),
        ),
    )


def scheduler_request(length_penalty: float) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(stage_payload=text_payload(length_penalty))
    )


def test_length_penalty_scales_eos_logits_per_request() -> None:
    original = torch.tensor(
        [
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
            [0.5, 2.0, -1.0, -4.0],
        ]
    )
    logits_output = SimpleNamespace(next_token_logits=original.clone())

    bare_runner().process_sampling_logits(
        logits_output,
        [
            scheduler_request(1.0),
            scheduler_request(2.0),
            scheduler_request(1.0),
        ],
    )

    torch.testing.assert_close(logits_output.next_token_logits[0], original[0])
    torch.testing.assert_close(logits_output.next_token_logits[2], original[2])
    expected = original[1].clone()
    expected[EOS_TOKEN_IDS] = torch.tensor([2.0 / 2.0, -4.0 * 2.0])
    torch.testing.assert_close(logits_output.next_token_logits[1], expected)


def test_length_penalty_disables_lookahead() -> None:
    runner = bare_runner()
    plain = SimpleNamespace(reqs=[schedule_request(1.0), schedule_request(1.0)])
    penalized = SimpleNamespace(
        reqs=[
            schedule_request(1.0),
            schedule_request(1.5),
        ]
    )

    assert runner.lookahead_eligible(plain) is True
    assert runner.lookahead_eligible(penalized) is False


def test_missing_thinker_length_penalty_is_neutral() -> None:
    assert resolve_thinker_length_penalty({}) == 1.0
    assert resolve_thinker_length_penalty({"stage_params": {}}) == 1.0
    assert (
        resolve_thinker_length_penalty({"stage_params": {"thinker": {"top_k": 20}}})
        == 1.0
    )
    assert (
        resolve_thinker_length_penalty(
            {"stage_params": {"thinker": {"length_penalty": None}}}
        )
        == 1.0
    )


def test_thinker_length_penalty_is_a_positive_float() -> None:
    params = {"stage_params": {"thinker": {"length_penalty": 1}}}
    assert resolve_thinker_length_penalty(params) == 1.0
    params = {"stage_params": {"thinker": {"length_penalty": 1.1}}}
    assert resolve_thinker_length_penalty(params) == 1.1


@pytest.mark.parametrize(
    "raw_penalty", [0, -1.0, True, "1.1", float("inf"), float("nan"), 10**1000]
)
def test_thinker_length_penalty_rejects_invalid_values(
    raw_penalty: int | float | bool | str,
) -> None:
    params = {"stage_params": {"thinker": {"length_penalty": raw_penalty}}}
    with pytest.raises(ValueError):
        resolve_thinker_length_penalty(params)
