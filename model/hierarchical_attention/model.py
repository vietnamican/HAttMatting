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
        low_level_feature,
        high_level_feature
    ):
        high_level_feature =  self.pyramidal_features_distillation(high_level_feature)
        return self.appearance_cue_filtration(low_level_feature, high_level_feature)
