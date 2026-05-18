# metashapes/lattice/basis.py

import torch
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Lattice:
    """In-plane periodicity of the unit cell. Fixed (non-optimizable).
    Defined by two lattice vectors. A rectangular lattice is the special
    case of axis-aligned vectors.

    Parameters
    ----------
    a1 : Tensor
        First lattice vector, shape [2].
    a2 : Tensor
        Second lattice vector, shape [2].
    """
    a1: torch.Tensor
    a2: torch.Tensor
    dtype: torch.dtype = field(default=torch.float32)
    device: str | torch.device = field(default="cpu")
    
    def __post_init__(self):
        a1 = torch.as_tensor(self.a1, dtype=self.dtype, device=self.device)
        a2 = torch.as_tensor(self.a2, dtype=self.dtype, device=self.device)
        object.__setattr__(self, "a1", a1)
        object.__setattr__(self, "a2", a2)
        
        if torch.linalg.det(self.matrix).abs().item() <= 1e-12:
            raise ValueError("a1 and a2 must be linearly independent")

    @classmethod
    def rectangular(cls, 
                    px: float, 
                    py: float,
                    dtype: torch.dtype = field(default=torch.float32),
                    device: str | torch.device = field(default="cpu")) -> "Lattice":
        """Build an axis-aligned rectangular lattice from two periods."""
        return cls(
            a1=torch.tensor([px, 0.0]),
            a2=torch.tensor([0.0, py]),
            dtype=dtype,
            device=device
        )

    # --- matrix form: columns are the lattice vectors ----------------
    @property
    def matrix(self) -> torch.Tensor:
        """2x2 matrix A with columns [a1 | a2]."""
        return torch.stack([self.a1, self.a2], dim=1)

    @property
    def cell_area(self) -> torch.Tensor:
        return torch.linalg.det(self.matrix).abs()

    # --- coordinate transforms ---------------------------------------
    def to_fractional(self, x: torch.Tensor, y: torch.Tensor):
        """Cartesian (x, y) -> fractional (f1, f2)."""
        p = torch.stack([x, y], dim=0)
        f = torch.einsum('ij,j...->i...', torch.linalg.inv(self.matrix), p)
        return f[0], f[1]

    def to_cartesian(self, f1: torch.Tensor, f2: torch.Tensor):
        """Fractional (f1, f2) -> Cartesian (x, y)."""
        f = torch.stack([f1, f2], dim=0)
        p = torch.einsum('ij,j...->i...', self.matrix, f)
        return p[0], p[1]

    # --- periodic copies ---------------------------------------------
    def offset(self, i: int, j: int) -> torch.Tensor:
        """Cartesian translation for lattice cell (i, j)."""
        return i * self.a1 + j * self.a2

    def neighbor_offsets(self, ring: int = 1):
        r = range(-ring, ring + 1)
        return [(i, j) for i in r for j in r]