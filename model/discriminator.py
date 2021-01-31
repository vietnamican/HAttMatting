from .base import Base


class Discriminator(Base):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        alpha_matte_true,
        alpha_matte_pred
    ):
        return x
