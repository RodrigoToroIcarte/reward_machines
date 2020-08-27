from gym.envs.registration import register

# ----------------------------------------- ANT

register(
    id='Ant-RM1-v0',
    entry_point='envs.mujoco_rm.ant_environment:MyAntEnvRM1',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Ant-RM4-v0',
    entry_point='envs.mujoco_rm.ant_environment:MyAntEnvRM4',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Ant-RM5-v0',
    entry_point='envs.mujoco_rm.ant_environment:MyAntEnvRM5',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Half-Cheetah-RM5-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM5',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id='Half-Cheetah-RM6-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM6',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM7-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM7',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM8-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM8',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM9-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM9',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM10-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM10',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM11-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM11',
    max_episode_steps=1000,
)



# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

# ----------------------------------------- CRAFT
for i in range(11):
    w_id = 'Craft-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )