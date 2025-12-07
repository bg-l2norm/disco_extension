from disco_rl.environments import jittable_envs
import ml_collections

def make_catch_env(batch_size: int, rows=5, columns=5):
    """
    Creates a Catch environment with configurable grid size.
    
    Args:
        batch_size: The number of environments to run in parallel.
        rows: The number of rows in the grid (default: 5).
        columns: The number of columns in the grid (default: 5).
        
    Returns:
        A Jittable Catch environment instance.
    """
    # Use the arguments provided in the function signature rather than hardcoded values.
    env_settings = ml_collections.ConfigDict(dict(rows=rows, columns=columns))
    
    return jittable_envs.CatchJittableEnvironment(
        batch_size=batch_size,
        env_settings=env_settings,
    )