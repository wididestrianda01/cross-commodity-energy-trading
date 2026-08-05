from pathlib import Path
from omegaconf import OmegaConf, DictConfig

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "pipeline.yaml"


def load_config(path: Path | None = None) -> DictConfig:
    p = path or CONFIG_PATH
    return OmegaConf.load(p)
