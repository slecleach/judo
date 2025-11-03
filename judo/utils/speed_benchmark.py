# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import time
from dataclasses import dataclass, field
from typing import List

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mujoco import MjData, MjModel, rollout

from judo import MODEL_PATH


def get_state(model: MjModel, data: list[MjData]) -> np.ndarray:
    """
    Get the state of the environment.

    Args:
        model: The mujoco model.
        data: The mujoco data.

    Returns:
        The state of the environment.

    Note:
        The state is a numpy array of shape (num_envs, state_size).
    """
    num_envs = len(data)
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(model, full_physics)
    state = np.zeros((num_envs, state_size), dtype=np.float64)
    for i in range(num_envs):
        mujoco.mj_getState(model, data[i], state[i], full_physics)
    return state


@dataclass
class SpeedBenchmarkConfig:
    """A class to configure the speed benchmark."""

    # model_path: str = field(default=str(MODEL_PATH / "xml/cartpole.xml"))
    model_path: str = field(default=str(MODEL_PATH / "xml/keith/scene.xml"))
    duration: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
    num_envs: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 48, 64])
    visualize: bool = field(default=False)


class SpeedBenchmark:
    """A class to benchmark the speed of a function."""

    def __init__(self, config: SpeedBenchmarkConfig) -> None:
        """
        Initialize the SpeedBenchmark class.

        Args:
            config: The configuration for the speed benchmark.
        """
        self.config = config
        self.model = MjModel.from_xml_path(config.model_path)
        self.rollout_ = rollout.Rollout(nthread=config.num_envs[0])
        self.max_num_envs = max(config.num_envs)
        self.data = [MjData(self.model) for _ in range(self.max_num_envs)]

        # Print model keyframe information
        if self.model.nkey > 0:
            print("\nModel Keyframes:")
            print(f"  Number of keyframes: {self.model.nkey}")
            print(f"  Key times: {self.model.key_time}")
            print(f"  First keyframe qpos: {self.model.key_qpos[0]}")
            print(f"  First keyframe qvel: {self.model.key_qvel[0]}")
            print(f"  First keyframe ctrl: {self.model.key_ctrl[0]}")
        else:
            print("\nModel has no keyframes defined.")

    def reset(self) -> None:
        """
        Reset the simulation.

        Note:
            The simulation is reset to the first keyframe.
        """
        assert self.model.nkey > 0, "Model has no keyframes defined."
        for i in range(self.max_num_envs):
            mujoco.mj_resetData(self.model, self.data[i])
            self.data[i].time = self.model.key_time[0]
            self.data[i].qpos[:] = self.model.key_qpos[0]
            self.data[i].qvel[:] = self.model.key_qvel[0]
            self.data[i].ctrl[:] = self.model.key_ctrl[0]
            mujoco.mj_forward(self.model, self.data[i])

    def update_data(self, state_rollout: np.ndarray) -> None:
        """
        Update the data with the last state of the rollout.

        Args:
            state_rollout: The state rollout. Shape=(num_envs, num_steps, state_dim).
        """
        num_envs = state_rollout.shape[0]
        # update the data with the last state of the rollout
        for i in range(num_envs):
            self.data[i].time = state_rollout[i, -1, 0]
            self.data[i].qpos[:] = state_rollout[i, -1, 1 : self.model.nq + 1]
            self.data[i].qvel[:] = state_rollout[i, -1, self.model.nq + 1 :]
            mujoco.mj_forward(self.model, self.data[i])

    def run(self, duration: float, num_envs: int) -> float:
        """
        Run the speed benchmark.

        Args:
            duration: Physics simulation duration in seconds.
            num_envs: Number of environments to run in parallel.

        Returns:
            Compute time in seconds.
        """
        self.reset()
        initial_state = get_state(self.model, self.data[:num_envs])  # (num_envs, state_size)
        num_steps = int(duration / self.model.opt.timestep)
        rollout_action = np.random.randn(num_envs, num_steps, self.model.nu)  # (num_envs, num_steps, action_size)

        rollout_ = rollout.Rollout(nthread=num_envs)

        # Time the rollout
        start_time = time.perf_counter()
        state_rollout, sensor_rollout = rollout_.rollout(
            self.model, self.data[:num_envs], initial_state, rollout_action
        )
        compute_time = time.perf_counter() - start_time

        self.update_data(state_rollout)
        return compute_time

    def benchmark_all(self) -> None:
        """
        Run benchmarks for all combinations of num_envs and duration.

        Computes and displays the real-time factor (physics_time / compute_time).
        """
        line_width = 120
        print(f"\n{'=' * line_width}")
        print("Speed Benchmark Results")
        print(f"Model: {self.config.model_path}")
        print(f"{'=' * line_width}")
        print(
            f"\n{'duration (s)':<15} {'num_envs':<12} {'compute_time (s)':<18} {'per_thread_real_time_factor':<30} {'real_time_factor':<25}"
        )
        print(f"{'-' * line_width}")

        results = []
        for duration in self.config.duration:
            print(f"\n{'─' * line_width}")
            print(f"Duration: {duration:.3f}s")
            print(f"{'─' * line_width}")
            for num_envs in self.config.num_envs:
                compute_time = self.run(duration, num_envs)
                per_thread_real_time_factor = duration / compute_time
                real_time_factor = per_thread_real_time_factor * num_envs

                results.append((num_envs, duration, compute_time, per_thread_real_time_factor, real_time_factor))

                print(
                    f"{duration:<15.3f} {num_envs:<12} {compute_time:<18.6f} {per_thread_real_time_factor:<30.2f}x {real_time_factor:<25.2f}x"
                )

        print(f"{'-' * line_width}\n")

    def visualize_trajectory(self, duration: float) -> None:
        """
        Visualize the trajectory of the robot using mujoco viewer.

        Args:
            duration: Physics simulation duration in seconds.
        """
        # Reset the simulation
        self.reset()
        data = self.data[0]  # Use first data instance for visualization

        num_steps = int(duration / self.model.opt.timestep)
        action = np.random.randn(1, num_steps, self.model.nu)

        # Create and launch the viewer
        with mujoco.viewer.launch_passive(self.model, data) as viewer:
            # Step through the simulation
            for step in range(num_steps):
                # Apply control action
                data.ctrl[:] = action[0, step, :]

                # Step the simulation
                mujoco.mj_step(self.model, data)

                # Sync viewer with simulation state
                viewer.sync()

                # Small delay to make visualization visible (optional)
                time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    config = tyro.cli(SpeedBenchmarkConfig)
    benchmark = SpeedBenchmark(config)
    benchmark.benchmark_all()
    if config.visualize:
        benchmark.visualize_trajectory(10.0)
