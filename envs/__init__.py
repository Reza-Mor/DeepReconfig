from gym.envs.registration import register

register(
    id="rnaenv-v1",
    entry_point="envs.gym_envs:Rnaenv_v1",
    kwargs={'dataset': 'datasets/expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
)

register(
    id="rnaenv-v2",
    entry_point="envs.gym_envs:Rnaenv_v2",
    kwargs={'dataset': 'datasets/expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
)

register(
    id="rnaenv-v3",
    entry_point="envs.gym_envs:Rnaenv_v3",
    kwargs={'dataset': 'datasets/expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
)

register(
    id="rnaenv-v4",
    entry_point="envs.gym_envs:Rnaenv_v4",
    kwargs={'dataset': 'datasets/expert_dbCRW_AND_entry_typeSequence_bonds_5by5'}
)

