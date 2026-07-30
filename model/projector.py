import mlx.nn as nn


class MlpProjector(nn.Module):
    def __init__(self, projector_type="linear", input_dim=2048, n_embed=1280):
        super().__init__()
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed

        if projector_type == "linear":
            self.layers = nn.Linear(input_dim, n_embed, bias=True)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

    def __call__(self, x):
        return self.layers(x)
