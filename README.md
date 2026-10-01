# Flexible Beam Controller-in-the-Loop Platform

基于数字孪生与控制器在环的柔性梁主动振动控制平台。

## 项目目标

本项目使用Python构建柔性梁、传感器和执行器数字孪生模型，
并使用面向STM32部署的C语言实现实时控制算法。

## 计划实现

- 柔性梁单自由度模型
- 柔性梁多模态模型
- 虚拟IMU及采样非理想因素
- 虚拟音圈执行器及输入饱和
- PID控制器
- 扰动观测器
- 残差元学习扰动估计器
- Python与C控制器一致性验证
- 故障注入与自动化测试
- 控制性能和执行时间评估

## 技术栈

- Python
- NumPy
- SciPy
- Matplotlib
- C
- CMake
- pytest
- Unity
- GitHub Actions

## 当前状态

项目初始化阶段。