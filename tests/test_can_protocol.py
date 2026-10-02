"""CAN通信协议测试。"""

import pytest

from communication.can_protocol import (
    CAN_PAYLOAD_LENGTH,
    CanFrameMonitor,
    CanProtocolError,
    CanStateMessage,
    calculate_crc8,
    decode_state_frame,
    encode_state_frame,
)


def test_encoded_frame_should_have_eight_bytes() -> None:
    message = CanStateMessage(
        displacement=0.0025,
        velocity=-0.125,
        sequence=10,
    )

    payload = encode_state_frame(message)

    assert len(payload) == CAN_PAYLOAD_LENGTH


def test_encode_and_decode_should_preserve_signals() -> None:
    message = CanStateMessage(
        displacement=0.002345,
        velocity=-0.125,
        sequence=42,
        measurement_valid=True,
        sensor_fault=True,
        actuator_saturated=True,
        deadline_missed=False,
    )

    decoded = decode_state_frame(encode_state_frame(message))

    assert decoded.displacement == pytest.approx(
        message.displacement,
        abs=0.000001,
    )

    assert decoded.velocity == pytest.approx(
        message.velocity,
        abs=0.001,
    )

    assert decoded.sequence == 42
    assert decoded.measurement_valid
    assert decoded.sensor_fault
    assert decoded.actuator_saturated
    assert not decoded.deadline_missed


def test_invalid_sequence_should_raise_error() -> None:
    message = CanStateMessage(
        displacement=0.0,
        velocity=0.0,
        sequence=256,
    )

    with pytest.raises(ValueError):
        encode_state_frame(message)


def test_corrupted_payload_should_fail_crc_check() -> None:
    message = CanStateMessage(
        displacement=0.001,
        velocity=0.2,
        sequence=1,
    )

    payload = bytearray(encode_state_frame(message))

    payload[0] ^= 0x01

    with pytest.raises(CanProtocolError):
        decode_state_frame(bytes(payload))


def test_invalid_payload_length_should_raise_error() -> None:
    with pytest.raises(CanProtocolError):
        decode_state_frame(b"\x00\x01")


def test_crc_should_be_repeatable() -> None:
    test_data = bytes([1, 2, 3, 4, 5, 6, 7])

    assert calculate_crc8(test_data) == calculate_crc8(test_data)


def test_monitor_should_count_lost_frames() -> None:
    monitor = CanFrameMonitor()

    first_payload = encode_state_frame(
        CanStateMessage(
            displacement=0.0,
            velocity=0.0,
            sequence=10,
        )
    )

    second_payload = encode_state_frame(
        CanStateMessage(
            displacement=0.0,
            velocity=0.0,
            sequence=13,
        )
    )

    monitor.receive(
        payload=first_payload,
        timestamp=0.0,
    )

    result = monitor.receive(
        payload=second_payload,
        timestamp=0.01,
    )

    assert result.valid
    assert result.lost_frames == 2
    assert result.total_lost_frames == 2


def test_sequence_rollover_should_not_count_loss() -> None:
    monitor = CanFrameMonitor()

    monitor.receive(
        payload=encode_state_frame(
            CanStateMessage(
                displacement=0.0,
                velocity=0.0,
                sequence=255,
            )
        ),
        timestamp=0.0,
    )

    result = monitor.receive(
        payload=encode_state_frame(
            CanStateMessage(
                displacement=0.0,
                velocity=0.0,
                sequence=0,
            )
        ),
        timestamp=0.01,
    )

    assert result.lost_frames == 0
    assert result.total_lost_frames == 0


def test_monitor_should_count_invalid_frame() -> None:
    monitor = CanFrameMonitor()

    payload = bytearray(
        encode_state_frame(
            CanStateMessage(
                displacement=0.0,
                velocity=0.0,
                sequence=1,
            )
        )
    )

    payload[2] ^= 0x40

    result = monitor.receive(
        payload=bytes(payload),
        timestamp=0.0,
    )

    assert not result.valid
    assert result.message is None
    assert result.invalid_frame_count == 1


def test_monitor_should_detect_timeout() -> None:
    monitor = CanFrameMonitor()

    assert monitor.is_timed_out(
        timestamp=0.0,
        timeout=0.02,
    )

    monitor.receive(
        payload=encode_state_frame(
            CanStateMessage(
                displacement=0.0,
                velocity=0.0,
                sequence=1,
            )
        ),
        timestamp=0.01,
    )

    assert not monitor.is_timed_out(
        timestamp=0.02,
        timeout=0.02,
    )

    assert monitor.is_timed_out(
        timestamp=0.04,
        timeout=0.02,
    )


def test_monitor_reset_should_clear_diagnostics() -> None:
    monitor = CanFrameMonitor()

    monitor.receive(
        payload=encode_state_frame(
            CanStateMessage(
                displacement=0.0,
                velocity=0.0,
                sequence=1,
            )
        ),
        timestamp=0.0,
    )

    monitor.reset()

    assert monitor.last_sequence is None
    assert monitor.last_valid_timestamp is None
    assert monitor.total_lost_frames == 0
    assert monitor.invalid_frame_count == 0
