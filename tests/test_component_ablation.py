from aegisland.component_ablation import run_matrix


def _by_name():
    result = run_matrix()

    return {
        variant.name: variant
        for variant in result.variants
    }


def test_semantic_trust_eliminates_unsafe_continuation() -> None:
    variants = _by_name()

    assert (
        variants["raw_baseline"]
        .unsafe_continuation_frames
        > 0
    )

    assert (
        variants["semantic_trust"]
        .unsafe_continuation_frames
        == 0
    )


def test_semantic_trust_reaches_safe_state_immediately() -> None:
    variants = _by_name()

    assert (
        variants["semantic_trust"]
        .time_to_safe_state_frames
        == 0
    )


def test_full_stack_preserves_zero_unsafe_continuation() -> None:
    variants = _by_name()

    assert (
        variants["pre_stabilizer_stack"]
        .unsafe_continuation_frames
        == 0
    )

    assert (
        variants["full_aegisland"]
        .unsafe_continuation_frames
        == 0
    )


def test_full_aegisland_recovers_no_earlier_than_pre_stabilizer() -> None:
    variants = _by_name()

    assert (
        variants["full_aegisland"]
        .recovery_frame
        >=
        variants["pre_stabilizer_stack"]
        .recovery_frame
    )
