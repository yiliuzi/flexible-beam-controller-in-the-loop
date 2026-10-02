"""支持延迟、丢包和数据损坏的虚拟CAN总线。"""

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VirtualCanParameters:
    """虚拟CAN总线参数。"""

    transmission_delay: float = 0.001
    drop_probability: float = 0.0
    corruption_probability: float = 0.0
    seed: int = 2028

    def __post_init__(self) -> None:
        if self.transmission_delay < 0:
            raise ValueError("transmission_delay must not be negative")

        if not 0.0 <= self.drop_probability <= 1.0:
            raise ValueError("drop_probability must be between 0 and 1")

        if not 0.0 <= self.corruption_probability <= 1.0:
            raise ValueError("corruption_probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CanTransmissionResult:
    """一次CAN发送操作的结果。"""

    accepted: bool
    dropped: bool
    corrupted: bool
    disconnected: bool
    arrival_time: float | None


@dataclass(frozen=True, slots=True)
class QueuedCanFrame:
    """虚拟CAN总线中的待接收报文。"""

    arrival_time: float
    payload: bytes


class VirtualCanBus:
    """支持通信故障注入的虚拟CAN总线。"""

    def __init__(
        self,
        parameters: VirtualCanParameters | None = None,
    ) -> None:
        self.parameters = parameters or VirtualCanParameters()

        self._random = random.Random(self.parameters.seed)

        self._queue: deque[QueuedCanFrame] = deque()
        self._connected = True
        self._last_timestamp: float | None = None

        self.transmitted_frame_count = 0
        self.delivered_frame_count = 0
        self.dropped_frame_count = 0
        self.corrupted_frame_count = 0
        self.disconnected_frame_count = 0

    @property
    def connected(self) -> bool:
        """返回总线是否处于连接状态。"""

        return self._connected

    @property
    def pending_frame_count(self) -> int:
        """返回等待接收的报文数量。"""

        return len(self._queue)

    def set_connected(self, connected: bool) -> None:
        """连接或断开虚拟CAN总线。"""

        self._connected = connected

    def _validate_timestamp(
        self,
        timestamp: float,
    ) -> None:
        if timestamp < 0:
            raise ValueError("timestamp must not be negative")

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamp must be monotonically increasing")

        self._last_timestamp = timestamp

    def _corrupt_payload(self, payload: bytes) -> bytes:
        corrupted_payload = bytearray(payload)

        byte_index = self._random.randrange(len(corrupted_payload))
        bit_index = self._random.randrange(8)

        corrupted_payload[byte_index] ^= 1 << bit_index

        return bytes(corrupted_payload)

    def send(
        self,
        payload: bytes,
        timestamp: float,
    ) -> CanTransmissionResult:
        """向虚拟CAN总线发送一帧报文。"""

        self._validate_timestamp(timestamp)

        if not 1 <= len(payload) <= 8:
            raise ValueError("classic CAN payload must contain 1 to 8 bytes")

        self.transmitted_frame_count += 1

        if not self._connected:
            self.dropped_frame_count += 1
            self.disconnected_frame_count += 1

            return CanTransmissionResult(
                accepted=False,
                dropped=True,
                corrupted=False,
                disconnected=True,
                arrival_time=None,
            )

        if self._random.random() < self.parameters.drop_probability:
            self.dropped_frame_count += 1

            return CanTransmissionResult(
                accepted=False,
                dropped=True,
                corrupted=False,
                disconnected=False,
                arrival_time=None,
            )

        corrupted = self._random.random() < self.parameters.corruption_probability

        transmitted_payload = payload

        if corrupted:
            transmitted_payload = self._corrupt_payload(payload)
            self.corrupted_frame_count += 1

        arrival_time = timestamp + self.parameters.transmission_delay

        self._queue.append(
            QueuedCanFrame(
                arrival_time=arrival_time,
                payload=transmitted_payload,
            )
        )

        return CanTransmissionResult(
            accepted=True,
            dropped=False,
            corrupted=corrupted,
            disconnected=False,
            arrival_time=arrival_time,
        )

    def receive(
        self,
        timestamp: float,
    ) -> list[bytes]:
        """返回当前时刻已经到达的全部CAN报文。"""

        self._validate_timestamp(timestamp)

        received_payloads: list[bytes] = []

        while self._queue:
            frame = self._queue[0]

            if frame.arrival_time > timestamp + 1e-12:
                break

            self._queue.popleft()
            received_payloads.append(frame.payload)
            self.delivered_frame_count += 1

        return received_payloads

    def clear_pending_frames(self) -> None:
        """清除尚未到达接收端的报文。"""

        self._queue.clear()

    def reset(self) -> None:
        """复位总线状态、统计量和随机数发生器。"""

        self._random = random.Random(self.parameters.seed)

        self._queue.clear()
        self._connected = True
        self._last_timestamp = None

        self.transmitted_frame_count = 0
        self.delivered_frame_count = 0
        self.dropped_frame_count = 0
        self.corrupted_frame_count = 0
        self.disconnected_frame_count = 0
