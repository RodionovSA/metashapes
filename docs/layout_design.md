# Layout generator — design sketch & implementation plan

Status: **phases 1–3 implemented** · phase 4 open · see §11 for implementation notes

## 1. Goal

Add a non-periodic counterpart to `UnitCell`: a **`Layout`** that fills a finite
area (the *domain*) with many shapes according to placement rules. No
periodicity, no seam handling, no lattice copies.

Requirements (agreed in discussion):

| # | Requirement |
|---|---|
| R1 | Rules can be **random** (constrained packing, datasets) **or deterministic / field-driven** (shape parameters as a function of position, e.g. a phase map) |
| R2 | Shapes stay **fully inside** the domain |
| R3 | Designed for **many shapes** (10³–10⁵): spatial indexing from day one |
| R4 | **Differentiable** end to end: gradients reach shape parameters and positions |
| R5 | Domain is a regular `Shape`; the rendered result is `scene ∩ domain` |
| R6 | **Batched** generation with a report, like `GenerationBatchResult` |
| R7 | Isolated: existing `UnitCell` / generator / analyzer behavior unchanged |
| R8 | Generator outputs are **plain** (buffers only, no `nn.Parameter`). Learnability is assigned after generation |

### 1.1 Target scale

| Quantity | Typical value |
|---|---|
| Smallest particle | ~50 nm |
| Min gap (when not merged) | ~25 nm |
| Domain | a few µm up to hundreds of µm |
| Pixel size | ~5 nm (≈10 px across the smallest feature) |
| Atoms per layout | ~10⁴ (10 µm) … ~10⁶ (100–300 µm, pitch ~100–300 nm) |
| Full raster | 100 µm @ 5 nm = 20 000² = 4·10⁸ px ≈ 1.6 GB float32, before autograd |

Two consequences shape the design:
- **One whole-domain tensor is not viable** for differentiable work at the
  upper end, so tiled rendering is part of the core API (§4.4).
- **A per-atom Python loop does not scale to 10⁶.** `ShapeSet` stores atoms so
  that same-type atoms can be evaluated in a single vectorised call. The
  public API doesn't depend on that, so phase 1 can start with the simple
  loop without a later redesign (§4.1).

Coordinates reach ~10⁵–10⁶ nm. float32 resolves ~0.03–0.06 nm at that
magnitude, which is fine at 5 nm pixels. Tiles are still built relative to
the tile origin, so local evaluation doesn't lose precision.

## 2. Key observation

Only `UnitCell`, `UnitCellGenerator`, `_is_shape_valid` and
`analysis._compute_min_gap` know about periodicity. The `Shape` tree,
sdflib, booleans, transforms and the Shapely adapter are already
non-periodic. The samplers use the `Lattice` only for its bounding box
(`lattice_inner_bounds`). So the layout feature **reuses the shape layer
and the samplers unchanged** and adds a parallel stack beside the
periodic one.

## 3. Package layout

```
src/metashapes/shape/             # existing package, new files only
  shapeset.py      # ShapeSet: flat, spatially indexed container of shapes (a Shape)
  spatial.py       # SpatialHash (torch side): atom ↔ cell tables used by ShapeSet

src/metashapes/layout/            # new, self-contained
  __init__.py
  spatial.py       # GeometryHash: incremental Shapely-side index for generation / analysis
  layout.py        # Layout: domain + ShapeSet, rendering, tiling, checks, penalties
  sites.py         # placement strategies: where candidates go
  sources.py       # atom sources: what shape goes at a site
  config.py        # LayoutConfig
  generator.py     # LayoutGenerator, LayoutBatchResult
  analysis.py      # LayoutAnalyzer, LayoutMetrics
```

Changes to existing files (all additive):

| File | Change |
|---|---|
| `shape/__init__.py` | export `ShapeSet` (registered in `SHAPE_REGISTRY` via `@register_shape`, like every primitive) |
| `shape/utils.py` (or new `shape/learnable.py`) | `make_learnable(...)`, a generic post-hoc buffer → `nn.Parameter` conversion (§4.6) |
| `metashapes/__init__.py` | export `Layout` |
| `adapters/shapely/dispatch.py` | one branch: `ShapeSet` → `unary_union` of children |
| `adapters/yaml.py` | `save_layouts` / `load_layouts` / `save_layout_batch` / `load_layout_batch` |
| `analysis.py` *(optional)* | `_leaf_shapes` recurses into `ShapeSet`, only needed if a `ShapeSet` is used as a `UnitCell` scene |

Nothing existing imports `metashapes.layout`. `UnitCell`, the samplers,
`UnitCellAnalyzer` and the generators are not modified. `ShapeSet` lives in
`shape/` because it is a general container that `UnitCell` could also use as
a scene (for example in large supercells), not something specific to layouts.

Unrelated but should be fixed: `transform.py:18` does `from .lattice import Canvas`,
and `lattice` has no `Canvas`, so importing `metashapes.transform` likely fails today.

## 4. Core objects

### 4.1 `ShapeSet`: the scalable scene

A `Union` chain of N shapes costs O(N) per query point and recurses N levels
deep, so it is unusable at 10⁴ shapes. `ShapeSet` is a **flat `Shape`** holding
an `nn.ModuleList` of placed atoms. Its SDF is evaluated through a spatial index.

```python
class ShapeSet(Shape):
    def __init__(self, shapes: Sequence[Shape], *, halo: float, cell_size: float | None = None): ...

    shapes: nn.ModuleList          # each usually Translate(atom, dx, dy)
    halo: float                    # truncation distance (see below)

    def sdf(self, x, y) -> Tensor              # arbitrary points, via SpatialHash bucketing
    def sdf_grid(self, xs, ys) -> Tensor       # fast path for axis-aligned grids ("stamping")
    def bounds(self)                           # union of child bounds
    @property
    def min_feature_size(self)                 # min over children (None if any unknown)
    def _project(self)                         # delegates to children
    def rebuild_index(self, force=False)       # see 4.3
    def to_parametric(self) / from_parametric  # {"type": "ShapeSet", "halo":..., "shapes": [...]}
    def __len__, __iter__, __getitem__
```

**Truncated SDF.** Each atom's bounding box is padded by `halo` and
registered in the index. A query point outside every padded box is farther
than `halo` from all atoms, so returning `+halo` there is a valid lower bound
with the correct sign. Inside the halo the value is the exact min over nearby
atoms. Consequences:
- hard masks are exact;
- soft masks are exact while `halo ≫ softness`. The default is
  `halo = max(8·softness, min_gap)`, and it is a constructor argument;
- gradients are zero beyond the halo, which is correct because those atoms
  don't affect those pixels.

*(Halo semantics and options are under review: see §4.1.1.)*

**Storage designed for vectorisation.** Internally, atoms of the form
`Translate(Rotate(<primitive>))` with a supported primitive are grouped by
type, and their parameters are stacked into tensors (one sdflib call per
type per tile). Anything else, such as booleans or nested composites, goes to
a generic per-atom fallback list. Phase 1 may implement only the generic path.
The grouping is internal, so adding it later doesn't change the API.

Atoms combine with a **hard min** (no smooth blending between atoms).
Overlapping atoms therefore merge, which is the intended behavior when
`min_gap=None`.
Smooth or boolean composition *inside* an atom works as before.

**Grid path (stamping)**, used by `rasterize` / `mask`:
for each atom, compute the index window of its padded box on the grid,
evaluate `atom.sdf` on that sub-grid only, and combine all windows with one
`scatter_reduce(..., reduce="amin", include_self=True)` into a tensor
initialised to `+halo`. Cost is O(Σ window sizes) instead of O(N · pixels).
It is one autograd op, so the graph doesn't grow to N in-place slice writes.
*(Phase 1 verifies `amin` backward with `gradcheck`. The fallback is a
sequence of slice-wise `torch.minimum` writes.)*

**Point path**, used by `sdf(x, y)` for arbitrary points: hash points into
uniform cells, sort by cell id, and for each atom gather the points in the
cells it overlaps (`searchsorted`). Evaluate, then `scatter_reduce` back.

Bonus: a `ShapeSet` is a `Shape`, so it also works as a `UnitCell` scene,
for large supercells. That is not a goal, just not prevented.

#### 4.1.1 Halo: what it is and the options (pending decision)

**What it is.** The spatial index lets each pixel evaluate only the atoms
*near* it. "Near" means inside the atom's bounding box padded by `halo`. A
pixel outside every padded box isn't evaluated against any atom. All we know
is that its true distance is greater than `halo`, so we return `halo` there.
The SDF is exact up to `halo` and clamped beyond it.

**What depends on it:**

| Consumer | Needs exact distance up to | Halo needed |
|---|---|---|
| Hard mask (`d <= 0`) | sign only | ~1 pixel (the fill value must stay > 0) |
| Soft mask `σ(-d/s)` | a few `s`: at `d = 8s`, σ ≈ 3·10⁻⁴ | ≈ 8·softness |
| Gap / proximity penalties | `min_gap` | ≥ min_gap |
| Offset / dilate by `r` (via `d <= r`) | `r` | > r |
| SDF as data (ML input, visualisation) | everywhere | ∞ |

**What it costs.** Per atom, the work is the area of its padded window,
`(size + 2·halo)²`. With 50 nm atoms at 5 nm pixels:

| Halo | Window | Work vs bare bbox |
|---|---|---|
| 5 nm (1 px, hard mask) | 60² nm | 1.4× |
| 25 nm (= min_gap) | 100² nm | 4× |
| 40 nm (8·softness, s = 1 px) | 130² nm | 6.8× |
| ∞ (exact) | whole raster | O(N · pixels): unusable at scale |

At your scale the halo is therefore the main performance knob.

**Options:**
- **A. Stored, fixed.** `ShapeSet(shapes, halo=40)`, the same for every call.
  Simple, but one value has to serve both cheap hard masks and soft masks.
- **B. Stored and required, no default.** Explicit, but adds friction everywhere.
- **C. Per call, resolved automatically.** `mask()`/`rasterize()` take
  `halo=None`, which resolves from what the call needs: 1 px for a hard mask,
  8·softness for a soft one, `+inf` (exact) when asked for. A stored
  `ShapeSet.halo` only serves as the default for raw `sdf(x, y)` point queries
  and as a lower bound.
- **D. Always exact** (`halo=∞`). Correct everywhere, only feasible for small
  layouts or tests.

**Decision: C, with D available as `halo=math.inf`.**

**Defaults per entry point:**

| Call | Default halo | Rationale |
|---|---|---|
| `sdf(x, y)` (`ShapeSet` and `Layout`) | `math.inf`: exact | A method named `sdf` returns the distance field. Pass a finite `halo=` for the fast capped version |
| `rasterize()` | `math.inf`: exact | Returns distance values, so same reasoning as `sdf`. Finite `halo=` for speed |
| `mask(soft=False)` | 1 pixel | Only the sign matters |
| `mask(soft=True)` | 8·softness | Sigmoid tail is < 3·10⁻⁴ beyond it |
| penalties | passed explicitly (e.g. `min_gap`, `margin`) | Each needs a known range |

**Exact queries without O(N · pixels): ring search.** `halo=inf` is *not*
brute force. Points are bucketed into index cells. For each cell, rings of
neighbouring cells are visited in order of increasing distance, evaluating
the atoms registered there. The search stops when the next ring's minimum
distance exceeds the best distance found so far. Atoms are spread fairly
evenly over the area, so in populated regions this costs about as much as a
capped query; it only goes further in empty regions. Grid queries
(`rasterize`) run the same search per tile: stamp the near windows first,
then ring-search only the pixels that are still unresolved.

**Exactness caveats (accepted).** These hold even with `halo=inf`, and the
zero level set (the masks) is exact in every case:
1. *Merged atoms*: the hard min over atoms is exact outside, but inside an
   overlap region it underestimates the distance to the merged boundary,
   the same as the existing `Union`.
2. *Domain clip*: `max(scene, domain)` is a lower bound near the corners where
   atoms meet the domain boundary, the same as the existing `Intersection`.
   Generated layouts are fully contained, so the clip leaves them unchanged.
3. Atoms whose own SDF is a bound (`TShape`, `Difference`, …) keep that property. It costs nothing in
the index design, because the index stores *unpadded* boxes and the padding
is applied when windows are computed. So the halo can change per call without
rebuilding anything. Penalties that need distance (gap, containment) pass
their own halo, e.g. `halo=min_gap`.

### 4.2 `SpatialHash`

One uniform-grid index used in two places:
- **torch side** (inside `ShapeSet`): atom-id ↔ cell tables as tensors, so
  it works on GPU.
- **geometry side** (generation and analysis): an incremental dict
  `cell → [atom ids]` holding Shapely geometries, used for gap queries during
  random sequential addition.

Shapely's `STRtree` is static, so it suits analysis (`query(..., predicate="dwithin")`)
but not incremental insertion.

Default cell size: about the median padded atom bbox. Meta-atoms are similar
in size, so a uniform grid beats a tree here and is simple to keep on GPU.

### 4.3 Index staleness during optimization

Positions and sizes change during gradient descent, so the boxes registered
in the index become stale. This uses a Verlet-list scheme:
- boxes are padded by `halo + skin`;
- the set tracks the maximum bbox drift since the last build;
- it rebuilds the index when drift > `skin / 2`.

The index itself is discrete and non-differentiable, but it only decides
*which* atoms are evaluated where. The SDF values stay fully differentiable.
For the batched-parameter path in phase 4, box drift is computed from the
stacked parameter tensors, without a per-atom `.item()`.

### 4.4 `Layout`

```python
class Layout(nn.Module):
    def __init__(self, domain: Shape, scene: ShapeSet | Shape, *, clip: bool = True): ...

    domain: Shape        # finite bounds required (Rectangle, Ellipse, Difference for exclusion zones, ...)
    scene: ShapeSet

    # --- same names / semantics as UnitCell where they make sense ---
    def sdf(self, x, y)                      # max(scene, domain) if clip else scene
    def rasterize(self, nx=None, ny=None, *, pixel_size=None, bounds=None)
    def mask(self, nx=None, ny=None, *, soft=False, softness=None, pixel_size=None, bounds=None)
    def tiles(self, tile_size, *, pixel_size, overlap=0.0) -> Iterator[Tile]
        # Tile = (bounds, X, Y); the caller renders each with rasterize/mask(bounds=tile.bounds).
        # Differentiable at full scale by accumulating gradients over tiles:
        #   for t in layout.tiles(...): loss_t(layout.mask(bounds=t.bounds, ...)).backward()
    def extent(self)                         # (xmin, xmax, ymin, ymax) of domain bounds, imshow format
    def grid(self, ...)                      # X, Y used by rasterize
    def to_shapely(self)                     # unary_union(atoms) ∩ domain
    def to_parametric(self) / from_parametric

    # --- layout-specific ---
    @property
    def shapes(self)                         # the atoms
    def __len__(self)
    def check(self, *, margin=0.0, min_gap=None, min_feature_size=None) -> list[str]
    def violations(self, *, margin=0.0) -> list[int]   # indices of atoms not fully inside
    def containment_penalty(self, *, margin=0.0, softness=None, pixel_size=None) -> Tensor
```

- The grid is Cartesian, axis-aligned, over the domain bounds, and
  endpoint-excluded (same convention as `UnitCell`). `pixel_size=` is offered
  as well as `nx/ny`, since the user usually thinks in pixel size for large areas.
- `UnitCell`'s `repeat=` and `cartesian=` have no meaning here and are not offered.

### 4.5 Edge policy (R2)

The question was whether to clip, warn, or do something else. The
recommendation is three layers, one per situation:

| Situation | Mechanism | Why |
|---|---|---|
| **Generation** | Containment is a hard placement constraint: a candidate is rejected unless `domain.buffer(-margin).contains(atom)` (Shapely, with spatial hash) | Generated layouts never contain edge violations, so no clipping or warning is ever needed |
| **Rendering** (`clip=True`, default) | `sdf = max(scene_sdf, domain_sdf)`, i.e. `scene ∩ domain` as you proposed | Output (mask/GDS) can never exceed the domain, even mid-optimization |
| **Optimization** | `containment_penalty()`: `∫ σ(-d_scene/s) · relu(d_domain + margin) dA` over the raster, evaluated only for atoms whose boxes touch the domain's border band | Differentiable force that pushes atoms back inside, so clipping stays a safety net and not a design feature |
| **Hand-built layouts** | `check()` / `violations()` on demand; `Layout(..., strict=True)` raises on construction | No warning by default, because it would fire on every forward pass during optimization |

A clipped partial atom can violate `min_feature_size`, so `check()` reports
both "outside domain" and "clipped feature" as separate reasons.

### 4.6 Learnability is post-hoc (R8)

Generators only produce buffers. When a layout is to be optimized, the user
chooses afterwards what becomes learnable:

```python
from metashapes.shape import make_learnable

make_learnable(layout.scene, names=("dx", "dy"))                 # all positions
make_learnable(layout.scene, names=("axes",), select=[3, 17, 42]) # params of some atoms
make_learnable(cell.scene)                                       # also works on a UnitCell scene
```

`make_learnable(shape, names=None, select=None, recurse=True)` walks the
module tree and replaces each matching buffer with an `nn.Parameter` holding
the same value. `select` (atom indices or a predicate) only applies to a
`ShapeSet`. It is generic, so it lives in `shape/`, not `layout/`. For grouped
(vectorised) atoms, the stacked tensor becomes the Parameter. A partial
`select` uses a mask so that unselected rows get zero gradient.

## 5. Generation

### 5.1 Decomposition: *where* × *what* × *constraints*

```
Sites (where) ──► candidate (x, y) ──► Source (what) ──► atom at origin
                                                        │
                    Constraints (containment, gap, feature, size) ◄─ Translate(atom, x, y)
                                                        │
                                       accept → SpatialHash.insert / reject → retry
```

The existing samplers mix "what" and "where": they pick a position inside the
cell box. Over a large domain their default size ranges would also scale with
the domain. Instead, the layout generator calls a sampler with a small
**reference box** (`atom_box`, the meta-atom scale), recentres the result on
its bbox centre, and translates it to the site. The samplers,
`fixed_shape_params` and `shape_param_ranges` are reused unchanged.
`Bar` has infinite extent and is rejected in layout sources.

### 5.2 Placement strategies: `sites.py`

```python
class Sites(ABC):
    def candidates(self, domain: Shape, rng) -> Iterator[tuple[float, float]]: ...
    exhaustible: bool      # lattice sites finish; random ones are bounded by attempts / targets

LatticeSites(lattice: Lattice, origin=(0, 0), jitter=0.0)   # reuses Lattice: rect / hex / oblique grids
RandomSites()                                              # uniform in domain (rejection by domain sdf)
PoissonDiskSites(radius)                                   # Bridson; even random coverage  (phase 4)
ExplicitSites(points)                                      # user-supplied positions
```

- With `LatticeSites`, a rejected candidate leaves the site empty after
  `max_tries_per_site` source retries.
- With `RandomSites`, generation is **random sequential addition (RSA)**. It
  stops when it reaches `target_num_shapes`, reaches `target_fill_fraction`,
  or hits `max_consecutive_failures` (the saturation signal).

### 5.3 Atom sources: `sources.py`

```python
class AtomSource(ABC):
    def atom(self, x: float, y: float, rng) -> Shape: ...    # returned centred at origin

SamplerSource(atom_config: GeneratorConfig, atom_box: tuple[float, float])
    # random: existing samplers + allowed_shapes / weights / ranges, unchanged
ParametricSource(shape_cls, params: dict[str, value | callable(x, y) -> value])
    # field-driven: e.g. {"axes": lambda x, y: (r(x, y), r(x, y)), "angle": theta_map}
LibrarySource(field: callable(x, y) -> float | (array, extent), library: list[(value, Shape)],
              mode="nearest")                                  # phase 4: phase map → meta-atom
Regions([(region: Shape, source: AtomSource), ...], default: AtomSource | None)
    # sub-areas with different rules; membership = region.sdf(x, y) <= 0
```

R1 is covered by combining strategies and sources. Random packing is
`RandomSites + SamplerSource`. A field-driven layout is `LatticeSites + ParametricSource`
(or `LibrarySource`). Mixed setups use `Regions`.

`atom_config` deliberately **reuses `GeneratorConfig`**, which users already
know. Only its shape-related fields are read; the lattice fields are ignored
and documented as such.

### 5.4 `LayoutConfig` and `LayoutGenerator`

```python
@dataclass(slots=True)
class LayoutConfig:
    seed: int | None = None
    target_count: int = 1                       # layouts per batch (R6)

    # stopping
    target_num_shapes: int | None = None
    target_fill_fraction: float | None = None
    max_consecutive_failures: int = 1000
    max_tries_per_site: int = 10
    max_tries_per_layout: int = 3

    # constraints
    edge_margin: float = 0.0                    # min clearance from domain boundary
    min_gap: float | None = None                # None → atoms may intersect and merge
                                                # 0.0  → touching allowed, overlap not
    min_feature_size: float | None = None
    min_shape_size: float | None = None
    # no learnability flags: outputs are plain, see §4.6

    extra: dict[str, Any] = field(default_factory=dict)


class LayoutGenerator:
    def __init__(self, config: LayoutConfig, sites: Sites, source: AtomSource,
                 validator: LayoutValidator | None = None): ...
    def generate(self, domain: Shape) -> LayoutBatchResult: ...
    def generate_one(self, domain: Shape) -> Layout: ...


@dataclass(slots=True)
class LayoutBatchResult:
    layouts: list[Layout]
    report: GenerationReport                    # reused as is
    @classmethod
    def merge(cls, *results) -> "LayoutBatchResult": ...
```

The report reuses `GenerationReport` (its fields are generic). Layout-specific
statistics go in `parameter_ranges` / `metadata`: shapes per layout, fill
fraction, reasons sites were rejected, and the saturation flag.

Reproducibility: one `random.Random(seed)` drives sites and sources in a
fixed order, as in `UnitCellGenerator`.

## 6. Analysis: `analysis.py`

```python
@dataclass
class LayoutMetrics:
    n_shapes: int
    fill_fraction: float          # area(scene ∩ domain) / area(domain)
    min_gap: float | None         # STRtree nearest-neighbour, non-periodic
    min_feature_size: float | None
    min_edge_clearance: float     # distance of atoms to domain boundary
    n_outside: int
    n_clipped_features: int

class LayoutAnalyzer:
    def __init__(self, *, min_gap=None, min_feature_size=None, edge_margin=0.0): ...
    def metrics(self, layout) -> LayoutMetrics
    def check(self, layout) -> list[str]
    def validate(self, layout) -> str | None      # usable as a LayoutValidator
```

It reuses the private helpers in `metashapes/analysis.py` (`_leaf_shapes`,
`_to_geom`, `_bbox_size`) by import. The 8-image periodic gap logic is
**not** reused.

## 7. Serialization & export

- `ShapeSet.to_parametric()` → `{"type": "ShapeSet", "halo": h, "shapes": [...]}`.
  It is registered in `SHAPE_REGISTRY` when `metashapes.layout` is imported,
  which `metashapes/__init__` does.
- `Layout.to_parametric()` → `{"type": "Layout", "domain": ..., "scene": ..., "clip": ...}`.
- YAML functions are additive, and the existing `save_unit_cells`/`load_unit_cells` are untouched.
- For 10⁵–10⁶ atoms YAML is impractical. Phase 2 adds a columnar format
  (atoms grouped by type, parameters as arrays, `.npz`), with YAML kept for small layouts.
- GDS: `Layout.to_shapely()` → the existing `shapely_to_gdstk`. When `min_gap`
  was enforced, the atoms don't overlap, so an `union=False` export path can
  skip the slow `unary_union` of 10⁵ polygons.

## 8. Usage sketch

```python
from metashapes import Lattice
from metashapes.shape import Rectangle, Ellipse
from metashapes.generators import GeneratorConfig
from metashapes.layout import (Layout, LayoutGenerator, LayoutConfig,
                               RandomSites, LatticeSites, SamplerSource, ParametricSource)

domain = Ellipse(center=(0, 0), axes=(50_000, 50_000))          # 100 µm circular aperture

# --- random constrained packing (dataset) ---
gen = LayoutGenerator(
    LayoutConfig(seed=0, target_count=8, min_gap=80, edge_margin=200,
                 target_fill_fraction=0.35),
    sites=RandomSites(),
    source=SamplerSource(
        GeneratorConfig(allowed_shapes=("Rectangle", "Ellipse"), min_feature_size=60),
        atom_box=(400, 400),
    ),
)
batch = gen.generate(domain)            # LayoutBatchResult
layout = batch.layouts[0]
mask = layout.mask(pixel_size=20)

# --- field-driven (deterministic) on a hex grid ---
radius = lambda x, y: 60 + 80 * phase(x, y) / (2 * math.pi)
gen = LayoutGenerator(
    LayoutConfig(edge_margin=200),
    sites=LatticeSites(Lattice.hexagonal(400)),
    source=ParametricSource(Ellipse, {"axes": lambda x, y: (radius(x, y),) * 2}),
)
layout = gen.generate_one(domain)

# --- optimization (learnability assigned after generation) ---
make_learnable(layout.scene, names=("axes",))
opt = torch.optim.Adam(layout.parameters(), lr=1.0)
for _ in range(100):
    opt.zero_grad()
    for t in layout.tiles(10_000, pixel_size=5):           # tiled: full domain never in memory
        m = layout.mask(bounds=t.bounds, pixel_size=5, soft=True)
        objective(m, t).backward()
    (10.0 * layout.containment_penalty(margin=200)).backward()
    opt.step()
```

## 9. Implementation phases

Each phase ends with its tests passing and the existing test suite unchanged and green.

**Phase 1: core rendering (foundation for everything else)**
- `shape/spatial.py` `SpatialHash` and `shape/shapeset.py` `ShapeSet`
  (grid stamping + point path, per-call halo, exact ring-search path for `halo=inf`,
  storage ready for type grouping).
- `Layout` with `bounds=` windowed rendering and `tiles()`.
- Shapely dispatch branch, `to_parametric`/`from_parametric`, YAML save/load.
- Tests:
  - `ShapeSet.sdf` equals the `Union` chain within the halo, and is `+halo` outside;
  - with `halo=inf` (ring search) it equals the `Union` chain everywhere, including empty regions;
  - hard masks are identical;
  - `gradcheck` through stamping (params and positions);
  - `clip=True` equals `Intersection(scene, domain)`;
  - serialization round-trip.
- Benchmark: 10⁴ atoms on a 4096² raster, CPU and GPU if available.

**Phase 2: generation**
- `SpatialHash` (geometry side), `Sites` (`Lattice`, `Random`, `Explicit`),
  `SamplerSource`, `ParametricSource`, `Regions`, `LayoutConfig`,
  `LayoutGenerator`, `LayoutBatchResult`, columnar `.npz` save/load.
- Tests:
  - containment and `min_gap` hold, checked against brute-force Shapely on small cases;
  - same seed gives the same output;
  - batch report counts are correct;
  - the fill-fraction and saturation stopping rules work.

**Phase 3: analysis & optimization support**
- `LayoutAnalyzer`/`LayoutMetrics`, `check`/`violations`, `containment_penalty`,
  index rebuild with skin, `make_learnable`.
- Example notebook `examples/layout.ipynb`, covering random packing, a field-driven
  layout and a short optimization loop.

**Phase 4: scale & libraries**
- Vectorised evaluation of same-type atoms: stacked parameter tensors with
  one sdflib call per type, which removes the per-atom Python loop.
- `PoissonDiskSites`, `LibrarySource`, and GDS export without the union.

## 10. Decisions log

| # | Question | Decision |
|---|---|---|
| 1 | Overlaps when `min_gap=None` | Allowed: atoms intersect and merge. `min_gap=0` means touching but no overlap |
| 2 | Halo | **C + D**: resolved per call; `sdf`/`rasterize` exact by default (ring search), masks capped (1 px hard, 8·softness soft), penalties explicit (§4.1.1) |
| 2b | Exactness caveats | Accepted: merged-atom interiors and the domain clip give lower bounds, as with the existing `Union`/`Intersection`; masks always exact |
| 3 | Learnable outputs | Generators output plain buffers. `make_learnable(...)` after generation (§4.6) |
| 4 | Where `ShapeSet` lives | `shape/` |
| 5 | Scale | 50 nm atoms, 25 nm gaps, µm to hundreds of µm domains → tiling + vectorised storage (§1.1) |

Plan impact of 5: tiled rendering (`Layout.tiles`, `bounds=`) moves into
phase 1. Columnar `.npz` serialization moves to phase 2, because YAML for
10⁵–10⁶ atoms is impractical. Vectorised same-type evaluation stays in
phase 4, but its storage layout is fixed in phase 1.

## 11. Implementation notes (phases 1–3)

Where the code differs from, or pins down, the sketch above:

| Topic | What was implemented |
|---|---|
| `ShapeSet(halo=...)` | Optional, default `math.inf`. A finite stored halo is only the default for `sdf(x, y)` calls without `halo=` (e.g. nested in a `Union`). `mask`/`rasterize`/penalties pass their own halo. |
| Ring search | Implemented as an expanding-radius search: round *k* evaluates `min(sdf, r₀·2ᵏ)` only at still-unresolved points; values `< r` are exact. Once `r` covers the whole index extent the remaining points are brute-forced. Grids stamp first with `r₀ = cell_size`, then resolve leftovers this way. |
| Candidate pairs | `SpatialHash.pairs` is fully vectorised (sort points by cell key, one `searchsorted` per atom row segment). Only the atom SDF evaluation is a per-atom Python loop (phase 4 replaces it). One `scatter_reduce(amin)` per call; `gradcheck` passes through it. |
| Bound-SDF atoms | `Star`, `TShape`, `Difference`, … underestimate distance far away. Skipping atoms relies on "SDF ≥ distance to bbox", so `ShapeSet` can return a *larger* value than a `Union` chain there — still a lower bound on the true distance with the correct sign (tested). Hard masks are unaffected. |
| Index staleness | Rebuild check runs only when some atom has a learnable parameter (current boxes are recomputed per query). Buffers edited in place need `rebuild_index(force=True)`. |
| Registration | `ShapeSet` registers itself in `shape/shapeset.py` (not when `metashapes.layout` is imported, as §7 said). |
| `Tile` | Carries `bounds` (with overlap), `core`, `nx/ny`, and `rows/cols/crop` slices for stitching; `X`/`Y`/`xs`/`ys` are lazy properties. |
| `containment_penalty` | Needs `pixel_size` or `softness` (each defaults to the other). Border atoms are selected by bbox vs. the domain shrunk by `margin + 8·softness`; each is integrated on its own window, aligned with the domain pixel grid. |
| `Layout.project()` | Added; calls `scene.project_tree()`. `shape_to_shapely()` itself now projects the whole tree before converting (an optimizer step can leave a size negative until the next `sdf()`), so every Shapely path — layouts, `UnitCell`, analyzers, generators — is covered. |
| `min_gap=0` | Touching allowed, positive-area overlap rejected — in both the generator and `LayoutAnalyzer.check`. |
| Sources | `AtomSource.deterministic` (true for `ParametricSource`) disables `max_tries_per_site` retries that would repeat the same atom. `Regions` without `default` leaves out-of-region sites empty. |
| Report | `parameter_ranges = {num_shapes, fill_fraction}`; `metadata` holds per-layout stats (`n_shapes`, `fill_fraction`, `saturated`, `sites_tried`, `rejections`), aggregated `site_rejections` and the saturated count. |
| `n_clipped_features` | An atom cut by the domain boundary whose clipped piece is fully eroded by `threshold/2` (a sliver). Threshold is the analyzer's `min_feature_size`, else the atom's own. |
| Columnar format | `metashapes.adapters.npz.save_layouts_npz` / `load_layouts_npz`: atoms grouped by parametric structure, numeric leaves stacked per group, JSON header for domain/options. |
| `transform.py` | Broken `Canvas` import removed (duck-typed now); `gdstk`, `pya`, `skimage` imported lazily so the module imports without them. |

### Benchmark (phase 1)

`python benchmarks/layout_bench.py` — 10⁴ atoms (half ellipses, half rectangles, ~50 % of
cells occupied), 4096² raster:

| | CPU | GPU |
|---|---|---|
| `mask()` hard (halo 1 px) | 3.4 s | 9.4 s |
| `mask(soft=True)` (halo 8 px) | 3.9 s | 9.5 s |
| `rasterize()` exact | 5.4 s | 9.5 s |
| soft mask forward + backward | 12.8 s | 28.2 s |

Profiling shows >90 % of the time inside the atoms' own `sdf()` calls (dominated by the
ellipse closest-point solve), ~10⁴ small calls. On GPU that is pure kernel-launch overhead,
hence slower than CPU. Index/bookkeeping cost is small. This is exactly what phase 4's
grouped same-type evaluation removes (one sdflib call per type per tile).
