"""嵌入式控制系统故障安全状态机。"""

from dataclasses import dataclass
from enum import Enum, auto


class SafetyState(Enum):
    """系统安全运行状态。"""

    NORMAL = auto()
    DEGRADED = auto()
    SAFE_STOP = auto()
    RECOVERY = auto()


@dataclass(frozen=True, slots=True)
class SafetyParameters:
    """安全状态机切换参数。"""

    degraded_failure_count: int = 2
    safe_stop_failure_count: int = 5
    recovery_valid_frame_count: int = 3
    normal_valid_frame_count: int = 5
    degraded_force_scale: float = 0.50
    recovery_force_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.degraded_failure_count <= 0:
            raise ValueError("degraded_failure_count must be positive")

        if self.safe_stop_failure_count < self.degraded_failure_count:
            raise ValueError(
                "safe_stop_failure_count must not be smaller than degraded_failure_count"
            )

        if self.recovery_valid_frame_count <= 0:
            raise ValueError("recovery_valid_frame_count must be positive")

        if self.normal_valid_frame_count <= 0:
            raise ValueError("normal_valid_frame_count must be positive")

        if not 0.0 <= self.degraded_force_scale <= 1.0:
            raise ValueError("degraded_force_scale must be between 0 and 1")

        if not 0.0 <= self.recovery_force_scale <= 1.0:
            raise ValueError("recovery_force_scale must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SafetyInput:
    """安全状态机单次输入。"""

    communication_timed_out: bool = False
    valid_frame_received: bool = False
    sensor_fault: bool = False
    deadline_missed: bool = False
    emergency_stop_requested: bool = False


@dataclass(frozen=True, slots=True)
class SafetyOutput:
    """安全状态机单次输出。"""

    state: SafetyState
    actuator_enabled: bool
    force_scale: float
    state_changed: bool
    consecutive_failures: int
    consecutive_valid_frames: int


class FaultStateMachine:
    """管理正常、降级、停机和恢复状态。"""

    def __init__(
        self,
        parameters: SafetyParameters | None = None,
    ) -> None:
        self.parameters = parameters or SafetyParameters()

        self.state = SafetyState.RECOVERY
        self.consecutive_failures = 0
        self.consecutive_valid_frames = 0

    def _transition(
        self,
        new_state: SafetyState,
    ) -> bool:
        if new_state is self.state:
            return False

        self.state = new_state
        return True

    def _create_output(
        self,
        state_changed: bool,
    ) -> SafetyOutput:
        if self.state is SafetyState.NORMAL:
            actuator_enabled = True
            force_scale = 1.0

        elif self.state is SafetyState.DEGRADED:
            actuator_enabled = True
            force_scale = self.parameters.degraded_force_scale

        elif self.state is SafetyState.RECOVERY:
            actuator_enabled = True
            force_scale = self.parameters.recovery_force_scale

        else:
            actuator_enabled = False
            force_scale = 0.0

        return SafetyOutput(
            state=self.state,
            actuator_enabled=actuator_enabled,
            force_scale=force_scale,
            state_changed=state_changed,
            consecutive_failures=self.consecutive_failures,
            consecutive_valid_frames=(self.consecutive_valid_frames),
        )

    def update(
        self,
        safety_input: SafetyInput,
    ) -> SafetyOutput:
        """根据诊断信息更新安全状态。"""

        if safety_input.emergency_stop_requested:
            self.consecutive_failures = self.parameters.safe_stop_failure_count
            self.consecutive_valid_frames = 0

            changed = self._transition(SafetyState.SAFE_STOP)

            return self._create_output(changed)

        failure_detected = (
            safety_input.communication_timed_out
            or safety_input.sensor_fault
            or safety_input.deadline_missed
        )

        if failure_detected:
            self.consecutive_failures += 1
            self.consecutive_valid_frames = 0

        elif safety_input.valid_frame_received:
            self.consecutive_valid_frames += 1
            self.consecutive_failures = 0

        state_changed = False

        if self.consecutive_failures >= self.parameters.safe_stop_failure_count:
            state_changed = self._transition(SafetyState.SAFE_STOP)

        elif self.consecutive_failures >= self.parameters.degraded_failure_count:
            state_changed = self._transition(SafetyState.DEGRADED)

        elif self.state in {
            SafetyState.DEGRADED,
            SafetyState.SAFE_STOP,
        }:
            if self.consecutive_valid_frames >= self.parameters.recovery_valid_frame_count:
                state_changed = self._transition(SafetyState.RECOVERY)
                self.consecutive_valid_frames = 0

        elif self.state is SafetyState.RECOVERY:
            if self.consecutive_valid_frames >= self.parameters.normal_valid_frame_count:
                state_changed = self._transition(SafetyState.NORMAL)
                self.consecutive_valid_frames = 0

        return self._create_output(state_changed)

    def reset(self) -> None:
        """将状态机复位到受限恢复状态。"""

        self.state = SafetyState.RECOVERY
        self.consecutive_failures = 0
        self.consecutive_valid_frames = 0
