"""
Module: tests.unit.test_event_bus

Purpose:
Unit test suite for Asynchronous Event Bus (publish, subscribe, history filtering).
"""

import pytest
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent


def test_event_bus_publish_subscribe():
    bus = EventBus()
    received_events = []

    def subscriber_cb(event: SystemEvent):
        received_events.append(event)

    bus.subscribe(EventTopic.ORCHESTRATOR, subscriber_cb)

    event = SystemEvent(
        topic=EventTopic.ORCHESTRATOR,
        event_type=EventType.AUTONOMOUS_DECISION,
        source="unit_test",
        payload={"decision": "CONTINUE"},
    )

    bus.publish_sync(event)

    assert len(received_events) == 1
    assert received_events[0].event_type == EventType.AUTONOMOUS_DECISION
    assert received_events[0].payload["decision"] == "CONTINUE"


def test_event_bus_history_filtering():
    bus = EventBus()
    event = SystemEvent(
        topic=EventTopic.DRIFT,
        event_type=EventType.DRIFT_DETECTED,
        source="unit_test",
        payload={"mmd": 0.15},
    )
    bus.publish_sync(event)
    history = bus.get_history(topic=EventTopic.DRIFT, limit=5)
    assert len(history) >= 1
    assert history[-1]["topic"] == "drift"
