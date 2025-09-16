# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.fields import np_1d_field

# TODOS
# add trace for second arm DONE
# add collision avoidance reward DONE
# speed up the physics DONE
# remove the gripper from both arms DONE
# add a watter jug DONE
# create a pick, lift and drop onto shelf task
# create a state machine for grasp object, use object to pull another object WONT DO
# fix friction DONE
# fix shelf DONE
# add proper water jug
# add stay in place term for the object during the homing phase

BOX_SIZE = 0.02
LOW_OBJ_HEIGHT = 0.125
XML_PATH = str(MODEL_PATH / "xml/fr3_bimanual_pick.xml")
QPOS_HOME = np.array(
    [
        0.4, 0, 0.2, 1, 0, 0, 0,  # object
        -0.5, 0.2, 0.0, -2.3562, 0.0, 1.5708, 0.7854,  # arm
        # 0.04, 0.04,  # gripper, equality constrained
        0.5, 0.2, 0.0, -2.3562, 0.0, 1.5708, 0.7854,  # arm_bis
        # 0.04, 0.04,  # gripper_bis, equality constrained
    ]
)  # fmt: skip
QPOS_TERMINAL = np.array(
    [
        0.4, 0, 0.2, 1, 0, 0, 0,  # object
        0, -np.pi/2, 0.0, -2.3562, 0.0, 1.5708, 0.7854,  # arm
        # 0.04, 0.04,  # gripper, equality constrained
        0, -np.pi/2, 0.0, -2.3562, 0.0, 1.5708, 0.7854,  # arm_bis
        # 0.04, 0.04,  # gripper_bis, equality constrained
    ]
)  # fmt: skip


class Phase(Enum):
    """Defines the phases of the FR3 pick task."""

    LIFT = 0
    MOVE = 1
    PLACE = 2
    HOMING = 3


@slider("w_lift_close", 0.0, 10.0, 0.01)
@slider("w_lift_height", 0.0, 10.0, 0.01)
@dataclass
class LiftConfig:
    """Reward configuration for the lift phase of the FR3 pick task."""

    w_lift_close: float = 1.0
    w_lift_height: float = 10.0


@slider("w_move_goal", 0.0, 10.0, 0.01)
@slider("w_move_close", 0.0, 10.0, 0.01)
@dataclass
class MoveConfig:
    """Reward configuration for the move phase of the FR3 pick task."""

    w_move_goal: float = 5.0
    w_move_close: float = 5.0


@slider("w_place_table", 0.0, 10.0, 0.01)
@slider("w_place_goal", 0.0, 10.0, 0.01)
@dataclass
class PlaceConfig:
    """Reward configuration for the place phase of the FR3 pick task."""

    w_place_table: float = 1.0
    w_place_goal: float = 1.0


@slider("w_upright", 0.0, 10.0, 0.01)
@slider("w_coll", 0.0, 10.0, 0.01)
@slider("w_qvel", 0.0, 10.0, 0.01)
@slider("w_open", 0.0, 10.0, 0.01)
@dataclass
class GlobalConfig:
    """Global reward configuration for the FR3 pick task."""

    w_upright: float = 0.25
    w_coll: float = 0.1
    w_qvel: float = 0.005
    w_open: float = 2.0
    w_collision_avoidance: float = 125.0


@slider("goal_radius", 0.0, 0.75, 0.01)
@slider("pick_height", 0.0, 1.0, 0.01)
@slider("pick_height_tolerance", 0.0, 1.0, 0.01)
@dataclass
class FR3BimanualPickConfig(TaskConfig):
    """Reward configuration for FR3 pick task."""

    # reward weights
    lift_weights: LiftConfig = field(default_factory=LiftConfig)
    move_weights: MoveConfig = field(default_factory=MoveConfig)
    place_weights: PlaceConfig = field(default_factory=PlaceConfig)
    global_weights: GlobalConfig = field(default_factory=GlobalConfig)

    goal_pos: np.ndarray = np_1d_field(
        np.array([1.1, 0.0, 0.55]),
        names=["x", "y", "z"],
        mins=[0.4, -1.0, 0.0],
        maxs=[1.5, 1.0, 1.0],
        steps=[0.01, 0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )
    goal_radius: float = 0.25
    pick_height: float = 0.5
    pick_height_tolerance: float = 0.1


class FR3BimanualPick(Task[FR3BimanualPickConfig]):
    """Defines the FR3 pick task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the LEAP cube rotation task."""
        super().__init__(model_path, sim_model_path=sim_model_path)

        # object indices
        self.obj_pos_adr = self.get_joint_position_start_index("object_joint")
        self.obj_pos_slice = slice(self.obj_pos_adr, self.obj_pos_adr + 3)

        obj_vel_adr = self.get_joint_velocity_start_index("object_joint")
        self.obj_vel_slice = slice(obj_vel_adr, obj_vel_adr + 3)

        obj_angvel_adr = obj_vel_adr + 3
        self.obj_angvel_slice = slice(obj_angvel_adr, obj_angvel_adr + 3)

        # robot indices
        arm_pos_adr = self.get_joint_position_start_index("fr3_joint1")
        # self.arm_pos_slice = slice(arm_pos_adr, arm_pos_adr + 9)  # 7 + 2 dofs for the gripper
        self.arm_pos_slice = slice(arm_pos_adr, arm_pos_adr + 7)  # 7 + 2 dofs for the gripper

        arm_bis_pos_adr = self.get_joint_position_start_index("_fr3_joint1")
        # self.arm_bis_pos_slice = slice(arm_bis_pos_adr, arm_bis_pos_adr + 9)  # 7 + 2 dofs for the gripper
        self.arm_bis_pos_slice = slice(arm_bis_pos_adr, arm_bis_pos_adr + 7)  # 7 + 2 dofs for the gripper
        self.arms_pos_slice = slice(arm_pos_adr, arm_pos_adr + 14)

        self.reset_command = QPOS_HOME[self.arms_pos_slice]

        # sensors
        # self.left_finger_obj_adr = self.get_sensor_start_index("left_finger_obj")
        # self.right_finger_obj_adr = self.get_sensor_start_index("right_finger_obj")
        # self.left_finger_table_adr = self.get_sensor_start_index("left_finger_table")
        # self.right_finger_table_adr = self.get_sensor_start_index("right_finger_table")
        self.grasp_site_adr = self.get_sensor_start_index("trace_grasp_site")
        self.ee_z_adr = self.get_sensor_start_index("ee_z")
        self.ee_z_slice = slice(self.ee_z_adr, self.ee_z_adr + 3)

        # self.left_finger_bis_obj_adr = self.get_sensor_start_index("_left_finger_obj")
        # self.right_finger_bis_obj_adr = self.get_sensor_start_index("_right_finger_obj")
        # self.left_finger_bis_table_adr = self.get_sensor_start_index("_left_finger_table")
        # self.right_finger_bis_table_adr = self.get_sensor_start_index("_right_finger_table")
        self.grasp_site_bis_adr = self.get_sensor_start_index("_trace_grasp_site")
        self.ee_z_bis_adr = self.get_sensor_start_index("_ee_z")
        self.ee_z_bis_slice = slice(self.ee_z_bis_adr, self.ee_z_bis_adr + 3)

        self.obj_table_adr = self.get_sensor_start_index("obj_table")

        # collision_avoidance sensors
        self.arm_maximal_pos_slice = []
        self.arm_bis_maximal_pos_slice = []
        for sensor_name in [
            "fr3_link1_pos",
            "fr3_link2_pos",
            "fr3_link3_pos",
            "fr3_link4_pos",
            "fr3_link5_pos",
            "fr3_link6_pos",
            "fr3_link7_pos",
        ]:
            idx = self.get_sensor_start_index(sensor_name) + np.arange(3)
            self.arm_maximal_pos_slice.append(idx)

        for sensor_name in [
            "_fr3_link1_pos",
            "_fr3_link2_pos",
            "_fr3_link3_pos",
            "_fr3_link4_pos",
            "_fr3_link5_pos",
            "_fr3_link6_pos",
            "_fr3_link7_pos",
        ]:
            idx = self.get_sensor_start_index(sensor_name) + np.arange(3)
            self.arm_bis_maximal_pos_slice.append(idx)

        self.arm_maximal_pos_slice = np.concatenate(self.arm_maximal_pos_slice)
        self.arm_bis_maximal_pos_slice = np.concatenate(self.arm_bis_maximal_pos_slice)

        # metadata that stores the current phase of the task
        self._data = mujoco.MjData(self.model)  # used for computing hypothetical sensor data
        self.phase = Phase.LIFT  # default phase

        self.reset()

    def in_goal_xy(self, curr_state: np.ndarray, config: FR3BimanualPickConfig) -> np.ndarray:
        """Checks if the object is somewhere in the tube above the goal position of radius r.

        Args:
            curr_state: The current state value. Shape=(nq + nv,).
            config: The task configuration.

        Returns:
            in_goal: A bool indicating whether the object is in the goal region. Shape=(,).
        """
        obj_pos = curr_state[self.obj_pos_adr : self.obj_pos_adr + 2]  # (2,)
        dist = np.linalg.norm(obj_pos - config.goal_pos)
        in_goal = dist <= config.goal_radius
        return in_goal

    def check_sensor_dists(
        self,
        sensors: np.ndarray,
        pair: Literal[
            "left_finger_obj",
            "right_finger_obj",
            "left_finger_table",
            "right_finger_table",
            "left_finger_obj_bis",
            "right_finger_obj_bis",
            "left_finger_table_bis",
            "right_finger_table_bis",
            "obj_table",
        ],
    ) -> np.ndarray:
        """Computes the distance between a specified pair of bodies.

        Args:
            sensors: The sensor values. Shape=(num_rollouts, T, total_sensor_dim).
            pair: The pair of bodies to check contact for. One of "left_finger_obj", "right_finger_obj", or "obj_table".

        Returns:
            dist: An array indicating the distance between the specified pair. Shape=(num_rollouts, T).
        """
        if pair == "left_finger_obj":
            i = self.left_finger_obj_adr
        elif pair == "right_finger_obj":
            i = self.right_finger_obj_adr
        elif pair == "left_finger_table":
            i = self.left_finger_table_adr
        elif pair == "right_finger_table":
            i = self.right_finger_table_adr
        elif pair == "left_finger_obj_bis":
            i = self.left_finger_bis_obj_adr
        elif pair == "right_finger_obj_bis":
            i = self.right_finger_bis_obj_adr
        elif pair == "left_finger_table_bis":
            i = self.left_finger_bis_table_adr
        elif pair == "right_finger_table_bis":
            i = self.right_finger_bis_table_adr
        elif pair == "obj_table":
            i = self.obj_table_adr
        else:
            raise ValueError(
                f"Invalid pair: {pair}. Must be one of 'left_finger_obj', 'right_finger_obj', 'left_finger_table', 'right_finger_table', 'left_finger_obj_bis', 'right_finger_obj_bis', 'left_finger_table_bis', 'right_finger_table_bis', 'obj_table'."
            )
        dist = sensors[:, :, i]
        return dist

    def pre_rollout(self, curr_state: np.ndarray, config: FR3BimanualPickConfig) -> None:
        """Computes the current phase of the system."""
        # update the data object associated with the current state
        self._data.qpos[:] = curr_state[: self.model.nq]
        self._data.qvel[:] = curr_state[self.model.nq : self.model.nq + self.model.nv]
        mujoco.mj_forward(self.model, self._data)

        # BUG: mujoco distance sensor seems to be broken and doesn't always return signed distance, so here we instead
        # check the object z position
        # curr_sensor = self._data.sensordata  # (total_sensor_dim,)

        phase = Phase.LIFT  # default phase

        # check whether the phase is MOVE
        # obj_in_air = curr_sensor[self.obj_table_adr] > 0  # object is not touching the table
        obj_in_air = (
            curr_state[self.obj_pos_adr + 2] - config.pick_height >= -config.pick_height_tolerance
        )  # object z position is close to the pick height
        if obj_in_air:
            phase = Phase.MOVE  # if the object is in the air, we are in lift phase

        # # check whether the phase is PLACE
        # in_goal_xy = self.in_goal_xy(curr_state, config)
        # if in_goal_xy and obj_in_air:
        #     phase = Phase.PLACE  # if the object is in the goal xy, we are in place phase

        # check whether the phase is HOMING
        # obj_table_dist = curr_sensor[self.obj_table_adr]
        # # if in_goal_xy and obj_table_dist <= 0:
        # #     phase = Phase.HOMING
        # obj_z_pos = curr_state[self.obj_pos_adr + 2]  # z position of the object
        # if in_goal_xy and obj_z_pos <= BOX_SIZE + 1e-3:  # the cube is 4cm wide and we allow a tolerance

        obj_pos = curr_state[self.obj_pos_slice]
        # print("obj_pos", obj_pos)
        # print("config.goal_pos", config.goal_pos)
        # print("config.goal_radius", config.goal_radius)
        # print("norm", np.linalg.norm(obj_pos - config.goal_pos))
        if np.linalg.norm(obj_pos - config.goal_pos) <= config.goal_radius:
            phase = Phase.HOMING

        self.phase = phase
        print(f"Phase: {self.phase}")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: FR3BimanualPickConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the LEAP cube rotation tracking task reward.

        The reward function switches between 4 modes:
        * LIFT: The object is lifted from the table.
        * MOVE: The object is moved to the goal position.
        * PLACE: The object is placed on the table.
        * HOMING: The robot arm is returned to the home position.

        There are also global rewards that are always applied:
        * Upright: The end-effector is upright.
        * Collision: The robot hand is not touching the table.
        * Qvel: The robot arm is not moving too fast.
        """
        # querying sensors
        # left_finger_table_dist = self.check_sensor_dists(sensors, "left_finger_table")  # noqa: F841
        # right_finger_table_dist = self.check_sensor_dists(sensors, "right_finger_table")  # noqa: F841
        # obj_table_dist = self.check_sensor_dists(sensors, "obj_table")  # noqa: F841
        # grasp_site_pos = sensors[..., self.grasp_site_adr : self.grasp_site_adr + 3]  # (num_rollouts, T, 3)
        # ee_z_axis = sensors[..., self.ee_z_slice]  # (num_rollouts, T, 3)

        # collision avoidance
        arm_maximal_pos = sensors[..., self.arm_maximal_pos_slice]  # (num_rollouts, T, 7*3)
        arm_bis_maximal_pos = sensors[..., self.arm_bis_maximal_pos_slice]  # (num_rollouts, T, 7*3)
        # reshape to (num_rollouts, T, 7, 3)
        arm_maximal_pos = arm_maximal_pos.reshape(arm_maximal_pos.shape[0], arm_maximal_pos.shape[1], 7, 3)
        arm_bis_maximal_pos = arm_bis_maximal_pos.reshape(
            arm_bis_maximal_pos.shape[0], arm_bis_maximal_pos.shape[1], 7, 3
        )
        # tile and compute pairwise distances
        tiled_arm_maximal_pos = np.tile(
            np.expand_dims(arm_maximal_pos, axis=2), (1, 1, 7, 1, 1)
        )  # (num_rollouts, T, 1, 7, 3)
        tiled_arm_bis_maximal_pos = np.tile(
            np.expand_dims(arm_bis_maximal_pos, axis=3), (1, 1, 1, 7, 1)
        )  # (num_rollouts, T, 7, 1, 3)
        # compute the distance between the two arms
        arm_dist = np.linalg.norm(tiled_arm_maximal_pos - tiled_arm_bis_maximal_pos, axis=-1)  # (num_rollouts, T, 7, 7)
        arm_dist = np.mean(arm_dist, axis=(1, 2, 3))  # (num_rollouts,)
        # compute the reward for collision avoidance
        rew_collision_avoidance = -np.exp(-arm_dist)  # (num_rollouts,)

        # querying states
        obj_pos = states[..., self.obj_pos_slice]  # (num_rollouts, T, 3)
        arm_pos = states[..., self.arm_pos_slice]  # (num_rollouts, T, 9)
        arm_bis_pos = states[..., self.arm_bis_pos_slice]  # (num_rollouts, T, 9)
        # xy_pos = states[..., :2]  # (num_rollouts, T, 2)
        z_obj = states[..., self.obj_pos_adr + 2]  # (num_rollouts, T)
        qvel = states[..., self.model.nq : self.model.nq + self.model.nv]  # (num_rollouts, T, nv)
        qvel_norm = np.linalg.norm(qvel, axis=-1)  # (num_rollouts, T)
        # gripper_pos = arm_pos[..., -1]  # (num_rollouts, T)

        # distances and errors
        q_arm_goal = QPOS_TERMINAL[self.arm_pos_slice]  # (9,)
        q_arm_bis_goal = QPOS_TERMINAL[self.arm_bis_pos_slice]  # (9,)

        # grasp_dist = ((grasp_site_pos - obj_pos) ** 2).sum(-1)  # (num_rollouts, T)
        obj_lower_pos = obj_pos.copy() - np.array([0, 0, LOW_OBJ_HEIGHT])
        grasp_dist = np.linalg.norm(arm_maximal_pos[:, :, -1, :] - obj_lower_pos, axis=-1)  # (num_rollouts, T)
        grasp_dist_bis = np.linalg.norm(arm_bis_maximal_pos[:, :, -1, :] - obj_lower_pos, axis=-1)  # (num_rollouts, T)
        grasp_dist = grasp_dist + grasp_dist_bis
        pick_height_err = (z_obj - config.pick_height) ** 2  # (num_rollouts, T)
        obj_goal_pos_dist = np.linalg.norm(obj_pos - config.goal_pos, axis=-1)  # (num_rollouts, T)

        home_dist = np.linalg.norm(arm_pos - q_arm_goal, axis=-1) + np.linalg.norm(
            arm_bis_pos - q_arm_bis_goal, axis=-1
        )  # (num_rollouts, T)

        # contact checks
        # left_finger_touching = left_finger_table_dist <= 0.0  # (num_rollouts, T)
        # right_finger_touching = right_finger_table_dist <= 0.0  # (num_rollouts, T)
        # hand_touching = left_finger_touching | right_finger_touching

        # lift rewards
        if self.phase == Phase.LIFT:
            w_lift_close = config.lift_weights.w_lift_close
            w_lift_height = config.lift_weights.w_lift_height
            rewards = -(w_lift_close * grasp_dist + w_lift_height * pick_height_err).sum(axis=-1)
            pass

        # move rewards
        elif self.phase == Phase.MOVE:
            w_move_goal = config.move_weights.w_move_goal
            w_move_close = config.move_weights.w_move_close
            rewards = -(w_move_goal * obj_goal_pos_dist + w_move_close * grasp_dist).sum(axis=-1)
            pass

        # place rewards
        elif self.phase == Phase.PLACE:
            # w_place_table = config.place_weights.w_place_table
            # w_place_goal = config.place_weights.w_place_goal
            # rewards = -(+w_place_table * obj_table_dist + w_place_goal * obj_goal_pos_dist).sum(axis=-1)
            pass

        # homing rewards
        elif self.phase == Phase.HOMING:
            rewards = -home_dist.sum(axis=-1)
            pass

        else:  # should never happen
            raise ValueError(f"Invalid phase: {self.phase}. Must be one of {list(Phase)}.")

        # global rewards
        # w_upright = config.global_weights.w_upright
        # w_coll = config.global_weights.w_coll
        w_qvel = config.global_weights.w_qvel
        # w_open = config.global_weights.w_open
        w_collision_avoidance = config.global_weights.w_collision_avoidance

        # rew_upright = -np.linalg.norm(ee_z_axis - np.array([[[0.0, 0.0, -1.0]]]), axis=-1).sum(axis=-1)
        # rew_coll = (1 - hand_touching).sum(axis=-1)  # (num_rollouts,)
        time_decay = np.linspace(1.0, 0.0, states.shape[1])  # decay the velocity penalty over time
        rew_qvel = -(time_decay * qvel_norm).sum(axis=-1)
        # rew_open = -((gripper_pos - 0.04) ** 2).sum(axis=-1)  # encourage the gripper to be open

        rewards += (
            # w_upright * rew_upright
            # + w_coll * rew_coll
            +w_qvel * rew_qvel
            # + w_open * rew_open
            + w_collision_avoidance * rew_collision_avoidance
        )
        return rewards

    def reset(self) -> None:
        """Resets the model to a default state with random goal."""
        self.data.qpos[:] = QPOS_HOME
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.reset_command
        mujoco.mj_forward(self.model, self.data)
