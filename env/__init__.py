from gymnasium.envs.registration import register

register(
    id="RobotArmPickAndPlace-v0",
    entry_point="env.pick_and_place:RobotArmPickAndPlaceEnv",
    max_episode_steps=150,
)

register(
    id="RobotArmPickAndPlaceMocap-v0",
    entry_point="env.pick_and_place_mocap:RobotArmPickAndPlaceMocapEnv",
    max_episode_steps=100,
)
