"""柔性梁控制系统CAN通信协议。"""

import struct
from dataclasses import dataclass

CAN_PAYLOAD_LENGTH = 8
PROTOCOL_VERSION = 1

DISPLACEMENT_SCALE = 0.000001
VELOCITY_SCALE = 0.001

STATUS_MEASUREMENT_VALID = 1 << 0
STATUS_SENSOR_FAULT = 1 << 1
STATUS_ACTUATOR_SATURATED = 1 << 2
STATUS_DEADLINE_MISSED = 1 << 3


class CanProtocolError(ValueError):
    """CAN报文格式或校验错误。"""


@dataclass(frozen=True, slots=True)
class CanStateMessage:
    """CAN总线传输的柔性梁状态。"""

    displacement: float
    velocity: float
    sequence: int
    measurement_valid: bool = True
    sensor_fault: bool = False
    actuator_saturated: bool = False
    deadline_missed: bool = False


@dataclass(frozen=True, slots=True)
class CanReceiveResult:
    """一次CAN报文接收结果。"""

    message: CanStateMessage | None
    valid: bool
    lost_frames: int
    total_lost_frames: int
    invalid_frame_count: int


def calculate_crc8(data: bytes) -> int:
    """使用多项式0x1D计算CRC-8校验值。"""

    crc = 0xFF

    for current_byte in data:
        crc ^= current_byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x1D) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def _encode_status(message: CanStateMessage) -> int:
    status = 0

    if message.measurement_valid:
        status |= STATUS_MEASUREMENT_VALID

    if message.sensor_fault:
        status |= STATUS_SENSOR_FAULT

    if message.actuator_saturated:
        status |= STATUS_ACTUATOR_SATURATED

    if message.deadline_missed:
        status |= STATUS_DEADLINE_MISSED

    return status


def encode_state_frame(message: CanStateMessage) -> bytes:
    """将柔性梁状态编码为8字节CAN数据。"""

    if not 0 <= message.sequence <= 255:
        raise ValueError("sequence must be between 0 and 255")

    displacement_raw = _clamp_int16(round(message.displacement / DISPLACEMENT_SCALE))

    velocity_raw = _clamp_int16(round(message.velocity / VELOCITY_SCALE))

    status = _encode_status(message)

    payload_without_crc = struct.pack(
        "<hhBBB",
        displacement_raw,
        velocity_raw,
        message.sequence,
        status,
        PROTOCOL_VERSION,
    )

    crc = calculate_crc8(payload_without_crc)

    return payload_without_crc + bytes([crc])


def decode_state_frame(payload: bytes) -> CanStateMessage:
    """解析并验证8字节CAN状态报文。"""

    if len(payload) != CAN_PAYLOAD_LENGTH:
        raise CanProtocolError("CAN payload must contain exactly 8 bytes")

    received_crc = payload[-1]
    calculated_crc = calculate_crc8(payload[:-1])

    if received_crc != calculated_crc:
        raise CanProtocolError("CAN payload CRC check failed")

    (
        displacement_raw,
        velocity_raw,
        sequence,
        status,
        protocol_version,
    ) = struct.unpack(
        "<hhBBB",
        payload[:-1],
    )

    if protocol_version != PROTOCOL_VERSION:
        raise CanProtocolError("unsupported CAN protocol version")

    return CanStateMessage(
        displacement=(displacement_raw * DISPLACEMENT_SCALE),
        velocity=velocity_raw * VELOCITY_SCALE,
        sequence=sequence,
        measurement_valid=bool(status & STATUS_MEASUREMENT_VALID),
        sensor_fault=bool(status & STATUS_SENSOR_FAULT),
        actuator_saturated=bool(status & STATUS_ACTUATOR_SATURATED),
        deadline_missed=bool(status & STATUS_DEADLINE_MISSED),
    )


class CanFrameMonitor:
    """监测CAN报文校验、丢帧和接收超时。"""

    def __init__(self) -> None:
        self.last_sequence: int | None = None
        self.last_valid_timestamp: float | None = None
        self.total_lost_frames = 0
        self.invalid_frame_count = 0

    def receive(
        self,
        payload: bytes,
        timestamp: float,
    ) -> CanReceiveResult:
        """接收CAN帧并更新通信诊断状态。"""

        if timestamp < 0:
            raise ValueError("timestamp must not be negative")

        try:
            message = decode_state_frame(payload)
        except CanProtocolError:
            self.invalid_frame_count += 1

            return CanReceiveResult(
                message=None,
                valid=False,
                lost_frames=0,
                total_lost_frames=self.total_lost_frames,
                invalid_frame_count=(self.invalid_frame_count),
            )

        lost_frames = 0

        if self.last_sequence is not None:
            sequence_delta = (message.sequence - self.last_sequence) % 256

            if sequence_delta > 1:
                lost_frames = sequence_delta - 1
                self.total_lost_frames += lost_frames

        self.last_sequence = message.sequence
        self.last_valid_timestamp = timestamp

        return CanReceiveResult(
            message=message,
            valid=True,
            lost_frames=lost_frames,
            total_lost_frames=self.total_lost_frames,
            invalid_frame_count=self.invalid_frame_count,
        )

    def is_timed_out(
        self,
        timestamp: float,
        timeout: float,
    ) -> bool:
        """判断距离最后一帧有效报文是否已经超时。"""

        if timestamp < 0:
            raise ValueError("timestamp must not be negative")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        if self.last_valid_timestamp is None:
            return True

        return timestamp - self.last_valid_timestamp > timeout

    def reset(self) -> None:
        """清除通信诊断状态。"""

        self.last_sequence = None
        self.last_valid_timestamp = None
        self.total_lost_frames = 0
        self.invalid_frame_count = 0
