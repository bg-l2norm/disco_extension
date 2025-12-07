from setuptools import setup, find_packages

setup(
    name="disco_extension",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "dm-haiku",
        "optax",
        "rlax",
        "distrax",
        "chex",
        "ml_collections",
        "pandas",
        "seaborn",
        "tqdm",
        "requests",
        # Note: disco_rl must be installed separately via git
    ],
    description="A user-friendly wrapper for Google DeepMind's Disco RL",
    author="Your Name",
)