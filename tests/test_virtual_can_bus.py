"""虚拟CAN总线测试。"""

import pytest

from communication.can_protocol import (
    CanStateMessage,
    decode_state_frame,
    encode_state_frame,
)
from communication.virtual_can_bus import (
    VirtualCanBus,
    VirtualCanParameters,
)


def create_payload(sequence: int = 1) -> bytes:
    """创建测试使用的CAN状态帧。"""

    return encode_state_frame(
        CanStateMessage(
            displacement=0.001,
            velocity=0.1,
            sequence=sequence,
        )
    )


def test_invalid_parameters_should_raise_error() -> None:
    with pytest.raises(ValueError):
        VirtualCanParameters(transmission_delay=-0.001)

    with pytest.raises(ValueError):
        VirtualCanParameters(drop_probability=1.1)

    with pytest.raises(ValueError):
        VirtualCanParameters(corruption_probability=-0.1)


def test_frame_should_arrive_after_configured_delay() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            transmission_delay=0.005,
        )
    )

    result = bus.send(
        payload=create_payload(),
        timestamp=0.0,
    )

    assert result.accepted
    assert result.arrival_time == pytest.approx(0.005)

    assert bus.receive(timestamp=0.004) == []

    received = bus.receive(timestamp=0.005)

    assert len(received) == 1
    assert received[0] == create_payload()


def test_zero_delay_frame_should_arrive_immediately() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            transmission_delay=0.0,
        )
    )

    payload = create_payload()

    bus.send(
        payload=payload,
        timestamp=0.0,
    )

    assert bus.receive(timestamp=0.0) == [payload]


def test_drop_probability_one_should_drop_every_frame() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            drop_probability=1.0,
        )
    )

    result = bus.send(
        payload=create_payload(),
        timestamp=0.0,
    )

    assert result.dropped
    assert not result.accepted
    assert bus.pending_frame_count == 0
    assert bus.dropped_frame_count == 1


def test_corruption_should_be_detected_by_crc() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            transmission_delay=0.0,
            corruption_probability=1.0,
        )
    )

    bus.send(
        payload=create_payload(),
        timestamp=0.0,
    )

    received = bus.receive(timestamp=0.0)

    assert len(received) == 1
    assert bus.corrupted_frame_count == 1

    with pytest.raises(ValueError):
        decode_state_frame(received[0])


def test_disconnected_bus_should_reject_frame() -> None:
    bus = VirtualCanBus()

    bus.set_connected(False)

    result = bus.send(
        payload=create_payload(),
        timestamp=0.0,
    )

    assert result.disconnected
    assert result.dropped
    assert not result.accepted

    assert bus.disconnected_frame_count == 1
    assert bus.pending_frame_count == 0


def test_receive_should_return_all_ready_frames() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            transmission_delay=0.002,
        )
    )

    first_payload = create_payload(sequence=1)
    second_payload = create_payload(sequence=2)

    bus.send(
        payload=first_payload,
        timestamp=0.0,
    )

    bus.send(
        payload=second_payload,
        timestamp=0.001,
    )

    assert bus.receive(timestamp=0.001) == []

    first_received = bus.receive(timestamp=0.002)
    second_received = bus.receive(timestamp=0.003)

    assert first_received == [first_payload]
    assert second_received == [second_payload]
    assert bus.delivered_frame_count == 2


def test_same_seed_should_reproduce_fault_sequence() -> None:
    parameters = VirtualCanParameters(
        transmission_delay=0.0,
        drop_probability=0.4,
        corruption_probability=0.3,
        seed=100,
    )

    first_bus = VirtualCanBus(parameters)
    second_bus = VirtualCanBus(parameters)

    first_results = []
    second_results = []

    for sequence in range(20):
        first_results.append(
            first_bus.send(
                payload=create_payload(sequence),
                timestamp=sequence * 0.001,
            )
        )

        second_results.append(
            second_bus.send(
                payload=create_payload(sequence),
                timestamp=sequence * 0.001,
            )
        )

    assert first_results == second_results


def test_decreasing_timestamp_should_raise_error() -> None:
    bus = VirtualCanBus()

    bus.send(
        payload=create_payload(),
        timestamp=0.01,
    )

    with pytest.raises(ValueError):
        bus.receive(timestamp=0.005)


def test_reset_should_clear_bus_state() -> None:
    bus = VirtualCanBus(
        VirtualCanParameters(
            transmission_delay=0.01,
        )
    )

    bus.send(
        payload=create_payload(),
        timestamp=0.0,
    )

    bus.set_connected(False)
    bus.reset()

    assert bus.connected
    assert bus.pending_frame_count == 0
    assert bus.transmitted_frame_count == 0
    assert bus.delivered_frame_count == 0
    assert bus.dropped_frame_count == 0
    assert bus.corrupted_frame_count == 0
    assert bus.disconnected_frame_count == 0
