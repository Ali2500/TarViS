from typing import Dict, Tuple, Union
from tarvis.utils.visualization import create_color_map
import numpy as np
import torch


class ColorGenerator:
    def __init__(self, semantic_color_map: Union[Dict[int, Tuple[int, int, int]], None], delta: int = 40):
        self.semantic_color_map = semantic_color_map
        self.cmap = create_color_map().tolist()
        self.delta = delta
        self._cache = {}
        self.rng = torch.Generator("cpu")
        self.rng.manual_seed(421987)

    def get_color(self, track_id: int, class_id: Union[int, None]):
        if track_id in self._cache:
            return self._cache[track_id]
        
        if self.semantic_color_map:
            assert class_id is not None
            base_color = torch.as_tensor(self.semantic_color_map[class_id])
            while True:
                offset = torch.randint(0, 30, (3,), generator=self.rng)
                color = (base_color + offset).clamp(0, 255).tolist()
                if color not in list(self._cache.values()):
                    break

            self._cache[track_id] = color
        else:
            self._cache[track_id] = self.cmap[track_id % 256]

        return self._cache[track_id]
