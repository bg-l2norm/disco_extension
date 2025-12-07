import jax
import jax.numpy as jnp
import chex
from disco_rl import types
from disco_rl import utils

def unroll_jittable_actor(
    params,
    actor_state,
    ts,
    env_state,
    rng,
    env,
    rollout_len,
    actor_step_fn,
):
  """Unrolls the policy for a jittable environment."""
  
  def _single_step(carry, step_rng):
    env_state, ts, actor_state = carry
    actor_timestep, actor_state = actor_step_fn(
        params, step_rng, ts, actor_state
    )
    env_state, ts = env.step(env_state, actor_timestep.actions)
    return (env_state, ts, actor_state), actor_timestep

  (env_state, ts, actor_state), actor_rollout = jax.lax.scan(
      _single_step,
      (env_state, ts, actor_state),
      jax.random.split(rng, rollout_len),
  )

  actor_rollout = types.ActorRollout.from_timestep(actor_rollout)
  return actor_rollout, actor_state, ts, env_state


def unroll_cpu_actor(
    params,
    actor_state,
    ts,
    env_state,
    rng,
    env,
    rollout_len,
    actor_step_fn,
    devices,
):
  """Unrolls the policy for a CPU environments."""
  actor_timesteps = []
  for _ in range(rollout_len):
    rng, step_rng = jax.random.split(rng)
    step_rng = jax.random.split(step_rng, len(devices))
    ts = utils.shard_across_devices(ts, devices)

    actor_timestep, actor_state = actor_step_fn(
        params, step_rng, ts, actor_state
    )
    actions = utils.gather_from_devices(actor_timestep.actions)
    env_state, ts = env.step(env_state, actions)

    actor_timesteps.append(actor_timestep)

  actor_rollout = types.ActorRollout.from_timestep(
      utils.tree_stack(actor_timesteps, axis=1)
  )
  return actor_rollout, actor_state, ts, env_state