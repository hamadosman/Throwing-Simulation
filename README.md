# Humanoid Ball Throw — Unitree G1 in MuJoCo

A reinforcement learning project where a full humanoid robot (the real-world **Unitree G1** platform) learns to throw a ball as far and as straight as possible inside a **MuJoCo** physics simulation. No hand-written motion rules — the policy is learned entirely through trial and error.

---

## Results

| Policy | Network | Reward | Outcome |
|--------|---------|--------|---------|
| 1 | MLP | Distance only | Throws sideways, reward hacking |
| 2 | LSTM | Distance + direction | Throws forward, releases immediately with no wind-up |
| 3 | LSTM | Distance + direction + domain randomization | Wind-up learned, ~50m best throw |

### Policy 1
<!-- To embed: drag Policy_1.mp4 into a GitHub Issue comment box, wait for upload, copy the generated URL, and paste it below -->
> 📹 *Video placeholder — see `RL training documentation/Policy_1.mp4`*

### Policy 2
<!-- To embed: drag Policy_2.mp4 into a GitHub Issue comment box, wait for upload, copy the generated URL, and paste it below -->
> 📹 *Video placeholder — see `RL training documentation/Policy_2.mp4`*

### Policy 3 (Final)
<!-- To embed: drag Policy_3_final.mp4 into a GitHub Issue comment box, wait for upload, copy the generated URL, and paste it below -->
> 📹 *Video placeholder — see `RL training documentation/Policy_3_final.mp4`*

---

## The problem

Throwing is deceptively hard for RL. The reward only arrives once the ball lands, so the agent has to connect a long sequence of body motions — wind-up, acceleration, release timing — to a single delayed signal. The robot also has many joints to coordinate simultaneously: waist, shoulder, elbow, and wrist all have to chain together correctly. A policy that just sees the current state with no memory of what led to it has a difficult time learning this.

---

## How training evolved

### Policy 1 — Baseline MLP, reward distance only

The first version used standard **PPO with a feedforward MLP**. The reward was simple: how far did the ball land from the starting position?

**What happened:** The agent found a local optimum immediately. It learned to throw **sideways** — a high-distance throw in the wrong direction scores the same as a forward throw under a pure distance reward. The agent had no incentive to aim, and no memory to connect early body motion to the eventual landing.

---

### Policy 2 — Recurrent PPO (LSTM) + directional reward

Two changes at once:

1. **Reward redesigned** — distance is now weighted by a directional factor. The further the ball lands from the forward centerline, the less reward the agent gets. Concretely: `reward = cos(angle/2) * distance`, so a perfectly sideways throw scores near zero regardless of distance.

2. **MLP replaced with LSTM** — switched to **RecurrentPPO** (`MlpLstmPolicy` from sb3-contrib). The LSTM gives the policy a short-term memory so it can connect the wind-up phase to the release timing and landing outcome.

**What happened:** The agent now throws forward. But it discovered a new shortcut — **release the ball immediately**, almost before any arm motion. Early release with no wind-up still gets some forward distance and the agent found this easier to learn than a full throwing motion. Progress, but the technique was wrong.

---

### Policy 3 — Domain randomization, forced wind-up

The fix: **randomize the starting joint angles on every episode reset**. At the start of each episode, all controlled joints are initialized to random positions within their joint limits.

This breaks the early-release shortcut. When the arm starts in a random configuration, immediately releasing the ball from wherever the arm happens to be produces inconsistent and mostly poor results. The agent is forced to learn to **move the arm into a good throwing position first** — which is exactly the wind-up behavior that was missing.

**What happened:** The agent learned a genuine wind-up and release sequence. Throw distance increased significantly, with the best observed throws reaching **~50 meters** in simulation.

---

## Key design details

**Environment (`throw_env.py`)**
- Controlled joints: waist (yaw/roll/pitch), right shoulder (pitch/roll/yaw), right elbow, right wrist (roll/pitch/yaw) — 10 joints total
- Actions: per-joint torques (scaled to actuator range) + one release signal
- Observations: joint positions and velocities, ball position and velocity, palm position and velocity, release flag
- Ball is attached to the right hand via a MuJoCo equality constraint (weld); `action[-1] > 0` drops the constraint and releases the ball
- Episode ends when the released ball hits the ground (`z ≤ 0.034`) or the time limit is reached (8 seconds)
- Pelvis is fixed so the agent focuses on arm and torso coordination, not locomotion

**Reward**
```
reward = cos(angle / 2) * actual_distance
```
where `angle` is the angle between the ball's landing direction and the robot's forward axis. A straight forward throw gets full distance credit; off-axis throws are penalized proportionally.

**Training (`train.py`)**
- Algorithm: `RecurrentPPO` with `MlpLstmPolicy`
- Network: `pi=[512, 256, 256]`, `vf=[256, 256, 256]`
- Learning rate: `1e-5`, clip range: `0.4`, entropy coefficient: `0.0`
- 4 parallel environments, 8M timesteps
- Checkpoints every 250k steps, best model saved by evaluation reward

---

## Repo structure

```
├── throw_env.py           # Gymnasium environment
├── train.py               # Training script
├── visualize.py           # Watch a loaded policy in the MuJoCo viewer
├── test.py                # Short evaluation script
├── sim.py                 # Inspect joint limits and model info
├── throw_policy.zip       # Saved policy weights (left arm fixed, ~50m best throw)
├── throw.xml              # Robot scene — place inside mujoco_menagerie/unitree_g1/
└── RL training documentation/
    ├── Policy_1.mp4
    ├── Policy_2.mp4
    ├── Policy_3_final.mp4
    └── Policy_3_Sample_Results.png
```

---

## Setup

**1. Clone MuJoCo Menagerie**

This project uses the Unitree G1 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Clone it into the project root:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
```

**2. Place the scene file**

Copy `throw.xml` into the Menagerie's G1 folder:

```bash
cp throw.xml mujoco_menagerie/unitree_g1/throw.xml
```

**3. Install dependencies**

```bash
pip install mujoco gymnasium stable-baselines3 sb3-contrib numpy
```

---

## Running

```bash
python3 train.py       # train from scratch
python3 visualize.py   # watch a saved policy (set model path in file)
```

---

## Compatibility note

`throw_policy.zip` was trained with the **left arm fixed** and the 10-joint observation/action layout above. If you modify `throw_env.py` or `throw.xml` the saved policy will not load — observation and action dimensions must match exactly. When in doubt, retrain.