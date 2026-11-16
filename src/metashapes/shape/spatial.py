# metashapes/shape/spatial.py
# Torch-side uniform-grid spatial index over axis-aligned boxes, used by
# ShapeSet to find which atoms can influence which query points.

from __future__ import annotations

import math

import torch

__all__ = ["SpatialHash", "ragged_arange"]


def ragged_arange(counts: torch.Tensor) -> torch.Tensor:
    """Concatenation of ``arange(c)`` for every ``c`` in ``counts``.

    ``ragged_arange(tensor([2, 0, 3])) -> tensor([0, 1, 0, 1, 2])``.
    """
    counts = counts.to(torch.long)
    total = int(counts.sum().item()) if counts.numel() else 0
    if total == 0:
        return torch.zeros(0, dtype=torch.long, device=counts.device)
    starts = torch.cumsum(counts, 0) - counts
    return torch.arange(total, device=counts.device) - torch.repeat_interleave(starts, counts)


class SpatialHash:
    """Uniform-grid index over a fixed set of axis-aligned boxes.

    The index stores the *unpadded* boxes. Every query passes its own
    padding (``pad``), so one index serves any halo without a rebuild: the
    padding is applied when the candidate pairs are computed.

    The index is discrete and non-differentiable. It only decides *which*
    atoms get evaluated at *which* points; the SDF values themselves are
    computed by the atoms, so gradients are unaffected.

    Parameters
    ----------
    boxes : Tensor [N, 4]
        ``(x0, y0, x1, y1)`` per atom. Rows with a non-finite or inverted
        box (empty or infinite atoms) are excluded from the index and never
        reported by ``pairs``.
    cell_size : float, optional
        Edge length of a grid cell. Defaults to the median of each box's
        longer side, which suits a population of similarly-sized meta-atoms.
    """

    def __init__(self, boxes: torch.Tensor, cell_size: float | None = None):
        boxes = torch.as_tensor(boxes, dtype=torch.float64).detach().cpu().reshape(-1, 4)
        self.boxes = boxes
        finite = torch.isfinite(boxes).all(dim=1)
        ordered = (boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])
        self.valid = finite & ordered
        self.ids = torch.nonzero(self.valid).flatten()          # index rows -> atom ids
        if cell_size is None:
            cell_size = self.default_cell_size(boxes[self.valid])
        if not (cell_size > 0 and math.isfinite(cell_size)):
            raise ValueError(f"cell_size must be positive and finite, got {cell_size}")
        self.cell_size = float(cell_size)
        self._cache: dict = {}

    @staticmethod
    def default_cell_size(boxes: torch.Tensor) -> float:
        if boxes.numel() == 0:
            return 1.0
        side = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        cs = float(side.median().item())
        if not (cs > 0 and math.isfinite(cs)):
            cs = float(side.max().item()) if side.numel() else 1.0
        return cs if cs > 0 and math.isfinite(cs) else 1.0

    def __len__(self) -> int:
        return int(self.ids.numel())

    def _boxes_on(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = (
                self.boxes[self.valid].to(device=device, dtype=dtype),
                self.ids.to(device=device),
            )
        return self._cache[key]

    def extent(self) -> tuple[float, float, float, float] | None:
        """``(x0, y0, x1, y1)`` covering every indexed box, or None if empty."""
        b = self.boxes[self.valid]
        if b.numel() == 0:
            return None
        return (b[:, 0].min().item(), b[:, 1].min().item(),
                b[:, 2].max().item(), b[:, 3].max().item())

    # ------------------------------------------------------------------
    def pairs(self, px: torch.Tensor, py: torch.Tensor, pad: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """All (atom, point) pairs where the point lies in the atom's box
        padded by ``pad``.

        ``px``/``py`` are flat [P] tensors. Returns ``(atom_ids, point_ids)``,
        both [M] long tensors on the points' device, grouped by atom id in
        ascending order.

        Cost is O(P log P + M): points are bucketed into grid cells and
        sorted by cell key once, then each atom's padded box is turned into
        one contiguous key range per grid row and resolved by
        ``searchsorted``. No per-atom Python loop.
        """
        if not math.isfinite(pad):
            raise ValueError("pairs() needs a finite pad; use brute force for pad=inf")
        device = px.device
        empty = (torch.zeros(0, dtype=torch.long, device=device),
                 torch.zeros(0, dtype=torch.long, device=device))
        if px.numel() == 0 or len(self) == 0:
            return empty

        # Index arithmetic in float64 regardless of the query dtype, so a
        # float32 raster far from the origin doesn't misplace cell borders.
        boxes, ids = self._boxes_on(device, torch.float64)
        qx = px.detach().to(torch.float64)
        qy = py.detach().to(torch.float64)
        cs = self.cell_size

        pix = torch.floor(qx / cs).to(torch.long)
        piy = torch.floor(qy / cs).to(torch.long)
        cx_lo, cx_hi = pix.min(), pix.max()
        cy_lo, cy_hi = piy.min(), piy.max()

        bx0, by0 = boxes[:, 0] - pad, boxes[:, 1] - pad
        bx1, by1 = boxes[:, 2] + pad, boxes[:, 3] + pad
        # Clamp the box extent to the points' cell range before casting to
        # long, so a huge pad can't overflow or produce enormous segments.
        lo_x = (cx_lo.to(torch.float64) - 1) * cs
        hi_x = (cx_hi.to(torch.float64) + 2) * cs
        lo_y = (cy_lo.to(torch.float64) - 1) * cs
        hi_y = (cy_hi.to(torch.float64) + 2) * cs
        ax0 = torch.floor(bx0.clamp(lo_x, hi_x) / cs).to(torch.long).clamp(min=cx_lo)
        ax1 = torch.floor(bx1.clamp(lo_x, hi_x) / cs).to(torch.long).clamp(max=cx_hi)
        ay0 = torch.floor(by0.clamp(lo_y, hi_y) / cs).to(torch.long).clamp(min=cy_lo)
        ay1 = torch.floor(by1.clamp(lo_y, hi_y) / cs).to(torch.long).clamp(max=cy_hi)

        live = (ax0 <= ax1) & (ay0 <= ay1)
        rows_live = torch.nonzero(live).flatten()
        if rows_live.numel() == 0:
            return empty

        W = cx_hi - cx_lo + 1
        key = (piy - cy_lo) * W + (pix - cx_lo)
        order = torch.argsort(key)
        skey = key[order]

        # One segment per (atom, grid row) the padded box spans.
        nrows = (ay1 - ay0 + 1)[rows_live]
        seg_row = torch.repeat_interleave(rows_live, nrows)
        row = ay0[seg_row] + ragged_arange(nrows)
        base = (row - cy_lo) * W
        k_lo = base + (ax0[seg_row] - cx_lo)
        k_hi = base + (ax1[seg_row] - cx_lo)
        start = torch.searchsorted(skey, k_lo, right=False)
        end = torch.searchsorted(skey, k_hi, right=True)
        lens = end - start

        pair_row = torch.repeat_interleave(seg_row, lens)
        pos = torch.repeat_interleave(start, lens) + ragged_arange(lens)
        pair_pt = order[pos]

        # Exact padded-box test (cells over-approximate the box).
        keep = ((qx[pair_pt] >= bx0[pair_row]) & (qx[pair_pt] <= bx1[pair_row])
                & (qy[pair_pt] >= by0[pair_row]) & (qy[pair_pt] <= by1[pair_row]))
        return ids[pair_row[keep]], pair_pt[keep]

    def grid_windows(self, xs: torch.Tensor, ys: torch.Tensor, pad: float
                     ) -> list[tuple[int, int, int, int, int]]:
        """Index windows of every padded box on an axis-aligned grid.

        ``xs`` [nx] and ``ys`` [ny] are ascending 1D sample coordinates.
        Returns ``(atom_id, i0, i1, j0, j1)`` for each atom whose padded
        box covers at least one sample: columns ``i0:i1``, rows ``j0:j1``.
        """
        if len(self) == 0 or xs.numel() == 0 or ys.numel() == 0:
            return []
        boxes, ids = self._boxes_on(xs.device, torch.float64)
        gx = xs.detach().to(torch.float64).contiguous()
        gy = ys.detach().to(torch.float64).contiguous()
        if math.isfinite(pad):
            bx0, by0 = boxes[:, 0] - pad, boxes[:, 1] - pad
            bx1, by1 = boxes[:, 2] + pad, boxes[:, 3] + pad
        else:
            n = boxes.shape[0]
            bx0 = by0 = torch.full((n,), -math.inf, dtype=torch.float64, device=xs.device)
            bx1 = by1 = torch.full((n,), math.inf, dtype=torch.float64, device=xs.device)
        i0 = torch.searchsorted(gx, bx0.contiguous(), right=False)
        i1 = torch.searchsorted(gx, bx1.contiguous(), right=True)
        j0 = torch.searchsorted(gy, by0.contiguous(), right=False)
        j1 = torch.searchsorted(gy, by1.contiguous(), right=True)
        live = (i1 > i0) & (j1 > j0)
        sel = torch.nonzero(live).flatten()
        if sel.numel() == 0:
            return []
        table = torch.stack([ids[sel], i0[sel], i1[sel], j0[sel], j1[sel]], dim=1)
        return [tuple(r) for r in table.cpu().tolist()]
