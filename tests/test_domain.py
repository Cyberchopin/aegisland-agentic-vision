import json

import pytest

from aegisland.domain import Telemetry, jsonable


def test_telemetry_rejects_impossible_values() -> None:
    with pytest.raises(ValueError):
        Telemetry(battery_percent=101, altitude_m=5)
    with pytest.raises(ValueError):
        Telemetry(battery_percent=50, altitude_m=-1)


def test_jsonable_output_can_be_serialized() -> None:
    payload = jsonable(Telemetry(50, 12, gps_available=False))
    assert json.loads(json.dumps(payload))["gps_available"] is False

