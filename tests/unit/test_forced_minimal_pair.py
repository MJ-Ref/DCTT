"""Unit tests for forced-token minimal-pair stress tests."""

from __future__ import annotations

from dctt.core.types import (
    DiagnosticResult,
    FrequencyTier,
    Stage1Result,
    Stage2Result,
    TokenInfo,
    TokenType,
)
from dctt.stress_tests.forced_minimal_pair import (
    ForcedTokenMinimalPairTest,
    build_minimal_pair_control_map,
)


def _make_result(
    token_id: int,
    token_str: str,
    severity: float,
) -> DiagnosticResult:
    token_info = TokenInfo(
        token_id=token_id,
        token_str=token_str,
        token_type=TokenType.CODE_SYMBOL,
        frequency=1.0,
        frequency_tier=FrequencyTier.MID,
        norm=1.0,
    )
    stage1 = Stage1Result(token_id=token_id, mu_k=0.1, med_k=0.1, spread_q=1.0)
    stage2 = Stage2Result(
        token_id=token_id,
        dim95=5,
        pr=5.0,
        cond=2.0,
        logdet=-1.0,
        anisotropy=1.2,
    )
    return DiagnosticResult(
        token_info=token_info,
        stage1=stage1,
        stage2=stage2,
        severity=severity,
        bucket=(FrequencyTier.MID, TokenType.CODE_SYMBOL),
    )


def test_build_minimal_pair_control_map_matches_bucket() -> None:
    """Control map uses same bucket with different token ids when possible."""
    results = [
        _make_result(1, "tok_a", severity=0.1),
        _make_result(2, "tok_b", severity=0.2),
        _make_result(3, "tok_c", severity=5.0),
        _make_result(4, "tok_d", severity=8.0),
    ]

    control_map = build_minimal_pair_control_map(results, seed=123)

    assert set(control_map.keys()) == {1, 2, 3, 4}
    for token_id, (control_id, _control_str) in control_map.items():
        assert control_id in {1, 2, 3, 4}
        if len(results) > 1:
            assert control_id != token_id


def test_forced_token_minimal_pair_run_single_records_control_gap() -> None:
    """run_single computes target and control failure rates for paired prompts."""
    stress_test = ForcedTokenMinimalPairTest(
        control_map={11: (22, "CTRL")},
        seed=7,
    )

    def model_fn(prompt: str) -> str:
        if "CTRL" in prompt:
            return "CTRL"
        if "GOOD" in prompt:
            return "mismatch"
        return ""

    result = stress_test.run_single(
        token_id=11,
        token_str="GOOD",
        model_fn=model_fn,
        n_cases=3,
    )

    assert result.failure_rate == 1.0
    assert result.details["control_failure_rate"] == 0.0
    assert result.details["failure_gap"] == 1.0


def test_forced_token_payload_handles_whitespace() -> None:
    """Whitespace tokens are escaped for reliable forced-token checks."""
    stress_test = ForcedTokenMinimalPairTest(control_map={})
    payload, escaped_mode = stress_test._token_payload(" \n")
    assert escaped_mode is True
    assert "\\" in payload


def test_forced_token_minimal_pair_logprob_scoring_path() -> None:
    """Logprob-choice mode should use score_options when available."""
    stress_test = ForcedTokenMinimalPairTest(
        control_map={11: (22, "CTRL")},
        scoring_mode="logprob_choice",
        min_logprob_margin=0.05,
    )

    class ChoiceModel:
        def __call__(self, prompt: str) -> str:
            _ = prompt
            return "unused"

        def score_options(self, prompt: str, options: list[str]) -> dict:
            expected = "CTRL" if "CTRL" in prompt else "GOOD"
            scores = {
                option: (0.0 if option == expected else -0.4)
                for option in options
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return {
                "best_option": ranked[0][0],
                "best_score": ranked[0][1],
                "margin": ranked[0][1] - ranked[1][1],
                "scores": scores,
            }

    result = stress_test.run_single(
        token_id=11,
        token_str="GOOD",
        model_fn=ChoiceModel(),
        n_cases=2,
    )

    assert result.failure_rate == 0.0
    assert result.details["control_failure_rate"] == 0.0
    assert result.details["failure_gap"] == 0.0
    assert result.details["scoring_mode"] == "logprob_choice"
    assert result.details["cases"][0]["scoring_mode"] == "logprob_choice"
    assert result.details["cases"][0]["target_margin"] is not None


def test_forced_token_minimal_pair_logprob_margin_enforced() -> None:
    """Cases fail in logprob mode when margin is below threshold."""
    stress_test = ForcedTokenMinimalPairTest(
        control_map={11: (22, "CTRL")},
        scoring_mode="logprob_choice",
        min_logprob_margin=0.5,
    )

    class LowMarginModel:
        def __call__(self, prompt: str) -> str:
            _ = prompt
            return "unused"

        def score_options(self, prompt: str, options: list[str]) -> dict:
            expected = "CTRL" if "CTRL" in prompt else "GOOD"
            scores = {
                option: (0.0 if option == expected else -0.1)
                for option in options
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return {
                "best_option": ranked[0][0],
                "best_score": ranked[0][1],
                "margin": ranked[0][1] - ranked[1][1],
                "scores": scores,
            }

    result = stress_test.run_single(
        token_id=11,
        token_str="GOOD",
        model_fn=LowMarginModel(),
        n_cases=3,
    )

    assert result.failure_rate == 1.0
    assert result.details["control_failure_rate"] == 1.0
    assert result.details["failure_gap"] == 0.0
