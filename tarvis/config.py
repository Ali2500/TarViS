from tarvis.utils.yaml_config import YamlConfig

import os

cfg = YamlConfig.load_from_file(os.path.realpath(os.path.join(
    os.path.dirname(__file__), os.pardir, "configs", "base.yaml"
)))

__all__ = [
    "cfg"
]
