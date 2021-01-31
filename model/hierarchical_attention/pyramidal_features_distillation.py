from ..base import Base

class PyramidalFeaturesDistillation(Base):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x
    ):
        return x