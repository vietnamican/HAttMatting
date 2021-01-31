from ..base import Base

class AppearanceCueFiltration(Base):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x
    ):
        return x