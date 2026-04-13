import numpy as np
import mujoco
import gymnasium as gym


class ThrowEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path('mujoco_menagerie/unitree_g1/throw.xml')
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)
 
        self.controlled_joints = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]


        self.ctrl_indices = []
        for name in self.controlled_joints:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.ctrl_indices.append(idx)

        self.weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, 'ball_weld')
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
        self.palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_middle_0_link')
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        self.model.eq_active0[self.weld_id] = 1
        self.data.eq_active[self.weld_id] = 1    

        n_ctrl = len(self.controlled_joints)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_ctrl + 1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * n_ctrl + 13,), dtype=np.float32
        )

        self.released = False
        self.max_time = 8.0

        mujoco.mj_forward(self.model, self.data)
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.initial_forward = self.data.xmat[torso_id].reshape(3, 3)[:, 0].copy()
        self.initial_ball_xy = self.data.xpos[self.ball_id][:2].copy()
        self.throw_origin = self.data.xpos[self.pelvis_id].copy()

        self.launch_angle = 0
        self.release_vel = 0
        self.reward_given = False

        self.reset_joint_angle_noise_std = np.radians(10)

    def reset(self, seed=None, options=None, random_reset = True):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed=None)

        mujoco.mj_resetData(self.model, self.data)
        self.model.eq_active0[self.weld_id] = 1
        self.data.eq_active[self.weld_id] = 1

        self.released = False

        if random_reset:
            for name in self.controlled_joints:
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr = self.model.jnt_qposadr[jnt_id]
                vadr = self.model.jnt_dofadr[jnt_id]
                lo, hi = self.model.jnt_range[jnt_id]
                q0 = float(self.data.qpos[qadr])

                self.data.qpos[qadr] = rng.uniform(lo,hi)
                self.data.qvel[vadr] = 0.0

        for i, ctrl_idx in enumerate(self.ctrl_indices):
            jnt_name = self.controlled_joints[i]
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            qadr = self.model.jnt_qposadr[jnt_id]
            self.data.ctrl[ctrl_idx] = self.data.qpos[qadr]

        mujoco.mj_forward(self.model, self.data)

        self.release_vel = 0.0
        self.launch_angle = 0
        self.reward_given = False
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for name in self.controlled_joints:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            obs.append(self.data.qpos[self.model.jnt_qposadr[jnt_id]])
            obs.append(self.data.qvel[self.model.jnt_dofadr[jnt_id]])

        obs.extend(self.data.xpos[self.ball_id])
        obs.extend(self.data.cvel[self.ball_id][3:])

        obs.extend(self.data.xpos[self.palm_id])
        obs.extend(self.data.cvel[self.palm_id][3:])

        obs.append(self.released)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        for i, ctrl_idx in enumerate(self.ctrl_indices):
            ctrl_range = self.model.actuator_ctrlrange[ctrl_idx]
            max_torque = max(abs(ctrl_range[0]), abs(ctrl_range[1]))
            self.data.ctrl[ctrl_idx] = action[i] * max_torque

        if action[-1] > 0 and not self.released:
            self.model.eq_active0[self.weld_id] = 0
            self.data.eq_active[self.weld_id] = 0
            self.released = True

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        ball_pos = self.data.xpos[self.ball_id]
        ball_vel = self.data.cvel[self.ball_id][3:]
        terminated = bool(self.released and ball_pos[2] <= 0.034)
        truncated = bool(self.data.time > self.max_time)

        reward = self._get_reward(ball_pos,ball_vel)

        return obs, reward, terminated, truncated, {}

    def _get_reward(self, ball_pos, ball_vel):
        reward = 0.0
        landed = self.released and ball_pos[2] <= 0.034

        if self.released and not self.reward_given:
            self.release_vel = np.linalg.norm(ball_vel)
            vz = ball_vel[2]
            vh = np.sqrt(ball_vel[0]**2 + ball_vel[1]**2)
            self.launch_angle = np.degrees(np.arctan2(vz, vh))
            self.reward_given = True 

        if landed:
            delta = ball_pos[:2] - self.initial_ball_xy
            fwd = self.initial_forward[:2]
            nd = np.linalg.norm(delta)
            nf = np.linalg.norm(fwd)
            if nd < 1e-8 or nf < 1e-8:
                angle = 0.0
            else:
                angle = np.arccos(np.clip(np.dot(delta, fwd) / (nd * nf), -1.0, 1.0))
            
            actual_dist = np.sqrt(
                (ball_pos[0] - self.throw_origin[0]) ** 2
                + (ball_pos[1] - self.throw_origin[1]) ** 2
            )

            angle_factor = np.cos(angle/2)
            
            
            """launch_factor = np.cos(np.radians(self.launch_angle-45))
            if self.launch_angle < -45 or self.launch_angle > 75:
               launch_factor = 0
            """
            reward =  angle_factor * actual_dist

        return reward
