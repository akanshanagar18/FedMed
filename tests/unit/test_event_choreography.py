"""
Module: tests.unit.test_event_choreography

Purpose:
Unit test suite for EventChoreographer (reactive event propagation from Drift and Health events).
"""

import pytest
from events.event_bus import global_event_bus, EventTopic, EventType, SystemEvent
from events.choreography import EventChoreographer, global_event_choreographer


def test_event_choreography_drift_reaction():
    event = SystemEvent(
        topic=EventTopic.DRIFT,
        event_type=EventType.DRIFT_DETECTED,
        source="unit_test",
        payload={"node_id": "hospital_alpha", "metrics": {"mmd": 0.18}},
    )
    global_event_bus.publish_sync(event)

    log = global_event_choreographer.choreography_log
    assert len(log) >= 1
    assert log[-1]["trigger_event"] == EventType.DRIFT_DETECTED.value
