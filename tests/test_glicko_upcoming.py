"""Booked bouts carry all-zero placeholder round rows; Glicko must still snapshot them."""
from app.services.ufc.glicko_service import _has_real_rounds


def test_placeholder_rounds_are_not_real():
    zero = {1: {11: {"kd": 0, "sig_str_landed": 0, "ctrl_seconds": 0},
                22: {"kd": 0, "sig_str_landed": 0, "ctrl_seconds": 0}}}
    assert not _has_real_rounds(zero)
    assert not _has_real_rounds({})
    assert not _has_real_rounds(None)


def test_played_rounds_are_real():
    played = {1: {11: {"kd": 0, "sig_str_landed": 14, "ctrl_seconds": 0},
                  22: {"kd": 0, "sig_str_landed": 9, "ctrl_seconds": 31}}}
    assert _has_real_rounds(played)
