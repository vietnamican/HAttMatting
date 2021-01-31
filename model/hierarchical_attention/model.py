from ..base import Base


class Model(Base):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        low_level,
        high_level
    ):
        return low_level
