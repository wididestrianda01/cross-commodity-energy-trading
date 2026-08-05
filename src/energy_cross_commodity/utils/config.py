from pathlib import Path
from omegaconf import DictConfig

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "pipeline.yaml"


def load_config(path: Path | None = None) -> DictConfig:
    """Load pipeline configuration from YAML via OmegaConf.

    Args:
        path: Optional path to a configuration file. Defaults to config/pipeline.yaml.

    Returns:
        OmegaConf DictConfig with typed data and portfolio sections.
    """
    from omegaconf import OmegaConf
    p = path or CONFIG_PATH
    return OmegaConf.load(p)
