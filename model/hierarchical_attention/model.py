from ..base import Base
from .appearance_cue_filtration import AppearanceCueFiltration
from .pyramidal_features_distillation import PyramidalFeaturesDistillation

class Model(Base):
    def __init__(self):
        super().__init__()
        self.appearance_cue_filtration = AppearanceCueFiltration()
        self.pyramidal_features_distillation = PyramidalFeaturesDistillation()

    def forward(
        self,
        low_level,
        high_level
    ):
        return self.pyramidal_features_distillation(high_level)
