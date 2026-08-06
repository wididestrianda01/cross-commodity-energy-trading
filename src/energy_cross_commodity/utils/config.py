from pathlib import Path
from omegaconf import DictConfig

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "pipeline.yaml"
ENV_PATH = CONFIG_PATH.parent.parent / ".env"


def load_config(path: Path | None = None) -> DictConfig:
    """Load pipeline configuration from YAML via OmegaConf.

    Also sources API credentials from a repository-root ``.env`` if present, so
    every entry point picks them up without the caller having to export them.
    Existing environment variables win, keeping CI and shell overrides in charge.

    Args:
        path: Optional path to a configuration file. Defaults to config/pipeline.yaml.

    Returns:
        OmegaConf DictConfig with typed data and portfolio sections.
    """
    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    load_dotenv(ENV_PATH, override=False)
    p = path or CONFIG_PATH
    return OmegaConf.load(p)
