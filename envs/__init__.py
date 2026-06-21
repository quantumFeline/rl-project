import gymnasium as gym

gym.register(
    id="NineRooms-v0",
    entry_point="envs.nine_rooms:NineRoomsEnv",
)

gym.register(
    id="DeceptiveRooms-v0",
    entry_point="envs.deceptive_rooms:DeceptiveRoomsEnv",
)
