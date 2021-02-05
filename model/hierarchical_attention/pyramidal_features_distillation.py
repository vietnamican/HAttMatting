from ..base import Base
from .cbam import ChannelGate

class PyramidalFeaturesDistillation(Base):
    def __init__(self):
        super().__init__()
        self.channel_gate = ChannelGate(256)
    
    def forward(
        self,
        x
    ):
        return self.channel_gate(x)
        