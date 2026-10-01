"""嵌入式周期控制任务和计算延迟模型。"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

ControllerCallback = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class ControlTaskParameters:
    """嵌入式控制任务时序参数。"""

    control_period: float = 0.005
    computation_delay: float = 0.001
    deadline: float = 0.005

    def __post_init__(self) -> None:
        if self.control_period <= 0:
            raise ValueError("control_period must be greater than zero")

        if self.computation_delay < 0:
            raise ValueError("computation_delay must not be negative")

        if self.deadline <= 0:
            raise ValueError("deadline must be greater than zero")


@dataclass(frozen=True, slots=True)
class ControlTaskOutput:
    """单个仿真时刻的控制任务输出。"""

    applied_command: float
    generated_command: float | None
    task_executed: bool
    command_activated: bool
    deadline_missed: bool
    execution_count: int
    deadline_miss_count: int


class PeriodicControlTask:
    """具有固定周期、延迟和零阶保持的控制任务。"""

    _TIME_TOLERANCE = 1e-12

    def __init__(
        self,
        controller: ControllerCallback,
        parameters: ControlTaskParameters | None = None,
    ) -> None:
        self.controller = controller
        self.parameters = parameters or ControlTaskParameters()

        self._next_release_time = 0.0
        self._last_timestamp: float | None = None
        self._applied_command = 0.0

        self._pending_commands: deque[tuple[float, float]] = deque()

        self.execution_count = 0
        self.deadline_miss_count = 0

    @property
    def applied_command(self) -> float:
        """返回当前通过零阶保持输出的控制指令。"""

        return self._applied_command

    @property
    def next_release_time(self) -> float:
        """返回下一次控制任务释放时间。"""

        return self._next_release_time

    def _activate_ready_commands(
        self,
        timestamp: float,
    ) -> bool:
        command_activated = False

        while self._pending_commands:
            activation_time, command = self._pending_commands[0]

            if activation_time > timestamp + self._TIME_TOLERANCE:
                break

            self._pending_commands.popleft()
            self._applied_command = command
            command_activated = True

        return command_activated

    def update(
        self,
        timestamp: float,
        measurement: float,
    ) -> ControlTaskOutput:
        """推进控制任务，并返回当前实际输出指令。"""

        if timestamp < 0:
            raise ValueError("timestamp must not be negative")

        if (
            self._last_timestamp is not None
            and timestamp < self._last_timestamp - self._TIME_TOLERANCE
        ):
            raise ValueError("timestamp must be monotonically increasing")

        self._last_timestamp = timestamp

        command_activated = self._activate_ready_commands(timestamp)

        task_executed = False
        generated_command: float | None = None
        deadline_missed = False

        if timestamp + self._TIME_TOLERANCE >= self._next_release_time:
            task_executed = True

            generated_command = self.controller(
                measurement,
                self.parameters.control_period,
            )

            activation_time = timestamp + self.parameters.computation_delay

            self._pending_commands.append((activation_time, generated_command))

            self.execution_count += 1

            deadline_missed = self.parameters.computation_delay > self.parameters.deadline

            if deadline_missed:
                self.deadline_miss_count += 1

            self._next_release_time += self.parameters.control_period

            command_activated = self._activate_ready_commands(timestamp) or command_activated

        return ControlTaskOutput(
            applied_command=self._applied_command,
            generated_command=generated_command,
            task_executed=task_executed,
            command_activated=command_activated,
            deadline_missed=deadline_missed,
            execution_count=self.execution_count,
            deadline_miss_count=self.deadline_miss_count,
        )

    def reset(self) -> None:
        """复位任务调度器和零阶保持输出。"""

        self._next_release_time = 0.0
        self._last_timestamp = None
        self._applied_command = 0.0
        self._pending_commands.clear()
        self.execution_count = 0
        self.deadline_miss_count = 0
