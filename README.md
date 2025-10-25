# Robot Arm Pick and Place

A reinforcement learning-based robot arm simulation environment and training code for pick-and-place tasks. This project uses the SAC (Soft Actor-Critic) algorithm to train a robot arm to complete object grasping tasks.

## Overview

This project implements a custom Gymnasium environment that simulates a single-side robotic arm with a suction gripper performing pick-and-place tasks. It uses the MuJoCo physics engine for simulation and implements the SAC reinforcement learning algorithm through the Stable-Baselines3 library.

### Key Features

- 🤖 Custom robot arm environment (based on Gymnasium)
- 🎯 Suction gripper simulation
- 📊 Comprehensive reward function design (distance, progress, action smoothness, etc.)
- 🎥 Training visualization and video recording
- 📈 TensorBoard logging

## Requirements

- Python 3.10+
- MuJoCo

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yichuan-huang/robot-arm-pick-and-place.git
cd robot-arm-pick-and-place
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Test Environment

Run the sample code to test if the environment works correctly:

```bash
python sample.py
```

This will open a visualization window where the robot arm performs random actions.

### 2. Train Model

Train the robot arm using the SAC algorithm:

```bash
python SAC_train.py
```

Training parameters:
- Total timesteps: 5,000,000
- Algorithm: SAC (Soft Actor-Critic)
- Policy: MultiInputPolicy
- Log directory: `./logs/SAC_pick_and_place/`
- Model save path: `./model/SAC_pick_and_place.zip`

### 3. Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir <logdir>
```

Then visit `http://localhost:6006` in your browser.

### 4. Test Trained Model

Run the test script and generate videos:

```bash
python SAC_play.py
```

This will load the trained model, run 50 test episodes, and record the results as a video saved in the `videos/` directory.

## Environment Details

### Observation Space

The environment uses a `Dict` observation space containing:
- `observation`: Robot joint states, end-effector position, object position, etc.
- `achieved_goal`: Currently achieved goal (object position)
- `desired_goal`: Desired goal position

### Action Space

4-dimensional continuous action space:
- Joint 1 rotation: [-1, 1] → mapped to actual angle changes
- Joint 2 rotation: [-1, 1]
- Joint 3 sliding: [-1, 1]
- Suction on/off: [-1, 1] (attempts to grip when > 0)

### Reward Function

Composite reward includes:
- **Progress reward**: Progress of object moving towards goal
- **Distance penalty**: Distance between object and goal
- **Action penalty**: Reduce unnecessary actions
- **Smoothness penalty**: Encourage smooth action sequences
- **Height reward**: Encourage lifting the object
- **Success reward**: Large reward when goal is reached

## TODO

- [ ] Complete model training until convergence
- [ ] Optimize reward function parameters
- [ ] Implement obstacle avoidance functionality
- [ ] Support objects with different shapes and weights
- [ ] Optimize suction gripper physics simulation
- [ ] Implement sim-to-real transfer for real robots
- [ ] Write unit tests


## License

MIT License
