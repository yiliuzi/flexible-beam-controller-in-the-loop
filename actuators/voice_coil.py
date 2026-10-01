"""虚拟音圈执行器模型。"""

from dataclasses import dataclass
from math import copysign, exp


@dataclass(frozen=True)
class VoiceCoilParameters:
    """音圈执行器参数。"""

    maximum_force: float = 2.0
    time_constant: float = 0.01
    dead_zone: float = 0.02

    def __post_init__(self) -> None:
        if self.maximum_force <= 0:
            raise ValueError("maximum_force 必须大于 0")

        if self.time_constant <= 0:
            raise ValueError("time_constant 必须大于 0")

        if not 0 <= self.dead_zone < self.maximum_force:
            raise ValueError("dead_zone 必须大于等于 0，并且小于 maximum_force")


@dataclass(frozen=True)
class VoiceCoilOutput:
    """单次执行器更新结果。"""

    requested_force: float
    target_force: float
    applied_force: float
    saturated: bool
    inside_dead_zone: bool
    enabled: bool


class VoiceCoilActuator:
    """带有饱和、死区和一阶惯性的音圈执行器。"""

    def __init__(
        self,
        parameters: VoiceCoilParameters | None = None,
    ) -> None:
        self.parameters = parameters or VoiceCoilParameters()
        self._applied_force = 0.0

    @property
    def applied_force(self) -> float:
        """返回当前实际输出力。"""

        return self._applied_force

    def reset(self) -> None:
        """清除执行器内部状态。"""

        self._applied_force = 0.0

    def emergency_stop(self) -> None:
        """立即将执行器输出清零。"""

        self._applied_force = 0.0

    def update(
        self,
        requested_force: float,
        time_step: float,
        *,
        enabled: bool = True,
    ) -> VoiceCoilOutput:
        """根据控制指令更新执行器实际输出力。"""

        if time_step <= 0:
            raise ValueError("time_step 必须大于 0")

        maximum_force = self.parameters.maximum_force
        dead_zone = self.parameters.dead_zone

        saturated = abs(requested_force) > maximum_force
        limited_force = max(
            -maximum_force,
            min(maximum_force, requested_force),
        )

        inside_dead_zone = enabled and abs(limited_force) <= dead_zone

        if not enabled or inside_dead_zone:
            target_force = 0.0
        else:
            effective_magnitude = (
                (abs(limited_force) - dead_zone) / (maximum_force - dead_zone) * maximum_force
            )
            target_force = copysign(
                effective_magnitude,
                limited_force,
            )

        response_ratio = 1.0 - exp(-time_step / self.parameters.time_constant)

        self._applied_force += response_ratio * (target_force - self._applied_force)

        self._applied_force = max(
            -maximum_force,
            min(maximum_force, self._applied_force),
        )

        return VoiceCoilOutput(
            requested_force=requested_force,
            target_force=target_force,
            applied_force=self._applied_force,
            saturated=saturated,
            inside_dead_zone=inside_dead_zone,
            enabled=enabled,
        )
