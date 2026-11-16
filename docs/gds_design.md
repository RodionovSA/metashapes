# GDS export — design

Status: **implemented (G1–G4)** · see §8 for implementation notes and deviations

GDS export for single shapes, `Layout`s, periodic `UnitCell` arrays, and an
assembly ("chip") that places many of them, with placement checks. This is the
GDS part of phase 4 in `layout_design.md`.

## 1. Decisions

| # | Topic | Decision |
|---|-------|----------|
| 1 | Conversion path | **Shape → Shapely → gdstk.** The Shapely adapters stay the single geometry source. Speed comes from GDS hierarchy (§4), not from a second set of converters. A direct gdstk fast path may be added later behind the same functions if profiling asks for it. |
| — | `transform.py` | **Obsolete; not used.** Its gdstk helpers are neither reused nor assumed correct or fast. The new code converts Shapely → gdstk itself (§3.2) and has its own tests. `transform.py` stays untouched. |
| 2 | Array edges | **Whole atoms.** An n₁×n₂ array is n₁·n₂ unclipped copies of the scene. The inside is exactly periodic, the outer edge is ragged, and nothing is clipped to the parallelogram. |
| 3 | Placement anchor | `Layout`: `at=(x, y)` moves the layout's own frame (domain coordinates are kept, and domains are usually centred on 0). Arrays: `anchor="center"` (default) puts the centre of the n₁a₁ × n₂a₂ parallelogram at `at`. `anchor="origin"` puts lattice point (0, 0) of the first cell there. |
| 4 | Units | GDS user unit **1 µm**, database precision **1 nm** by default. Model coordinates are unitless, and `coord_unit` (metres per model unit, default `1e-6`) says what they mean. A design drawn in nm passes `coord_unit=1e-9`. All lengths given to the GDS API (`at`, `tolerance`, `min_spacing`, …) are in **model units**. |
| 5 | Layers | One `(layer, datatype)` per placement, default `(0, 0)`. A placement does not split across layers. |
| 6 | Overlapping atoms | Exported **as they are**, as overlapping polygons with no union. |
| 7 | Text | No automatic labels. A placement may have an optional user `text=`, written as a `gdstk.Label` (an annotation that does not print) on a **separate text layer**, never on the design layer. Polygon text is opt-in (`text_as_polygons=True`) and still goes on the text layer. |

## 2. Module layout

```
src/metashapes/gds/            # needs gdstk (optional dependency, imported lazily)
  __init__.py                  # public API below
  convert.py                   # shapely -> gdstk polygons (holes, vertex limit, scaling)
  cells.py                     # shape_to_gdstk, unit_cell_to_cell, layout_to_cell
  assembly.py                  # GDSAssembly, Placement, footprints, checks, write
```

`metashapes/__init__` does not import `gds`, so the package keeps working
without gdstk. The Shapely-side change (§3.1) is in `adapters/shapely/`.

## 3. Conversion

### 3.1 Curve resolution in the Shapely adapters (prerequisite)

Today the number of segments on curves is fixed and differs between
converters:

| Where | Segments |
|---|---|
| Ellipse | 64 |
| Shapely buffers | 16 per quarter circle (Shapely's default) |
| `round_corners` | 32 per quarter circle |

For a 10 µm disc, 64 segments put the outline up to about 6 nm off the true
curve.

- `shape_to_shapely(shape, *, tolerance=None)`. `tolerance` is the largest
  allowed distance between a polygon edge and the true curve. It is passed to
  every converter that draws a curve.
- For an arc of radius r, the segments per quarter are
  `ceil((π/2) / (2·acos(1 − tol/r)))`, with at least 1 and a cap (e.g. 4096).
- Ellipse and Egg use their largest radius of curvature (`a²/b`) as r, which
  is conservative. `round_corners` uses the corner radius.
- `tolerance=None` keeps today's fixed counts, so analyzer results and
  existing tests don't change. GDS export always passes a tolerance
  (default `1e-3` µm = 1 nm, converted to model units).

### 3.2 Shapely → gdstk (new, `gds/convert.py`)

`polygons_from_shapely(geom, *, scale, layer, datatype) -> list[gdstk.Polygon]`

- Handles `Polygon`, `MultiPolygon` and `GeometryCollection`. Points and
  lines are dropped, and empty input gives `[]`.
- Multiplies coordinates by `scale = coord_unit / gds_unit`.
- **Holes:** GDS has none. A polygon with holes goes through
  `gdstk.boolean(exterior, holes, "not")`, which returns a single polygon with
  a cut line through to each hole. Only polygons that actually have holes pay
  for this.
- Drops pieces smaller than the database grid (area < precision²) so the
  writer doesn't get degenerate polygons.
- Vertex limit: `write_gds(max_points=8190)`. gdstk defaults to 199 and would
  split every smooth polygon into many pieces.

### 3.3 Objects → cells (`gds/cells.py`)

```python
shape_to_gdstk(shape, *, layer=0, datatype=0, tolerance, coord_unit=1e-6,
               unit=1e-6) -> list[gdstk.Polygon]

unit_cell_to_cell(cell: UnitCell, name, *, layer, datatype, tolerance, ...) -> gdstk.Cell
    # one period: the scene as drawn, unclipped (decision 2)

layout_to_cell(layout: Layout, name, *, layer, datatype, tolerance,
               dedupe=True, workers=None, ...) -> gdstk.Cell
```

`layout_to_cell` works as follows:

- **Containment.** It uses `Layout.violations()` to find the atoms that are
  not fully inside the domain.
  - Contained atoms are exported as they are.
  - Crossing atoms are intersected with the domain one by one (the domain is
    "prepared" once).
  - Atoms with nothing inside the domain are dropped.
  - With `clip=False` on the layout, nothing is clipped.
- **Deduplication (`dedupe=True`).** The key is the atom's `to_parametric()`
  with the outer `Translate` removed and floats rounded to the database grid.
  Dimensionless values (`angle`, `Scale.s`) are rounded to `grid / size`
  instead (size = the atom's extent), and lengths inside a `Scale` to
  `grid / |s|`, so two atoms share a key only if their outlines differ by
  less than about one grid step.
  - Each unique key becomes one sub-cell.
  - Each atom becomes a `gdstk.Reference` at its translation.
  - Keys used only once are written as plain polygons, so that layouts where
    every atom is unique (e.g. from `SamplerSource`) don't get one sub-cell
    per atom.
  - Clipped atoms are always plain polygons.
- **Parallel (`workers=N`).** This is phase G4 and only goes in if the
  benchmark shows it helps. Each worker receives the atom parameters (cheap to
  pickle), rebuilds the shapes, and returns vertex arrays. `gdstk` objects are
  created in the main process.

## 4. Hierarchy (where the speed comes from)

| Object | GDS structure | Cost |
|---|---|---|
| UnitCell array n₁×n₂ | one cell (one period) + one `Reference` with `Repetition(n₁, n₂, v1=a1, v2=a2)`; oblique lattices work | O(1) in n₁·n₂ |
| Layout, repeated atoms | one cell per unique atom + references | O(unique atoms) Shapely work |
| Layout, unique atoms | plain polygons | O(atoms) at ≈0.5 ms/atom with Shapely (≈50 s for 10⁵) |
| Same Layout / UnitCell placed twice | its cell is built once and referenced from each placement | — |

Measured today on 2000 random atoms (11 primitives):

| Step | Time per atom |
|---|---|
| `shape_to_shapely` | ≈480 µs |
| Shapely → gdstk | ≈30 µs |

## 5. Assembly

```python
from metashapes.gds import GDSAssembly

chip = GDSAssembly(name="TOP", unit=1e-6, precision=1e-9, coord_unit=1e-9,
                   tolerance=1.0, text_layer=(100, 0))

chip.place(layout, at=(0, 0), layer=(1, 0), name="lens_A", text="lens A, f=50um")
chip.place_array(unit_cell, at=(200_000, 0), repeats=(400, 400),
                 anchor="center", layer=(1, 0), name="grating_3")
chip.place_grid(items, origin=(0, -300_000), pitch=(150_000, 150_000), cols=8,
                repeats=(100, 100))          # batches: Layouts and/or UnitCells in a grid

chip.footprints() -> dict[name, shapely.Polygon]   # for plotting
chip.check(min_spacing=0.0) -> list[str]           # [] = OK
chip.to_library() -> gdstk.Library
chip.write_gds("chip.gds", allow_overlap=False)    # runs check(); raises on overlap
```

### 5.1 Placements

Each placement records: the item, its offset, the layer, a name (unique, and
used as the GDS cell name after sanitising), and an optional text.

- `place_array` needs `repeats=(n1, n2)`.
- `place_grid` is a convenience over `place` / `place_array`:
  - It accepts a `GenerationBatchResult`, a `LayoutBatchResult`, or a plain
    list.
  - Each item is centred in its grid slot (the footprint's bbox centre).
  - `repeats` applies to the UnitCells in the list.
  - Names are auto-generated (`{prefix}_{i}`), and texts can be passed as a
    list.
- The same object placed twice shares one cell (keyed by object identity).

### 5.2 Footprints

| Item | Footprint |
|---|---|
| Layout, `clip=True` | the domain polygon, moved |
| Layout, `clip=False` | the scene's bounding hull |
| Array | convex hull of the n₁a₁ × n₂a₂ parallelogram **and** the scene's bounding box placed at the four corner cells, so the ragged edge from decision 2 is covered |

### 5.3 Checks (`check`)

- **Overlap:** pairs of footprints whose intersection has area > 0. Touching
  is allowed.
- **Spacing:** pairs of footprints closer than `min_spacing`. Uses an STRtree
  with nearest-neighbour queries, so it works for thousands of placements.
- **Names:** duplicate placement names.
- The total extent is reported in the summary, but it is not a check.
- **Optional `check(geometry=True)`:** for pairs whose footprints overlap or
  come too close, also checks the real exported polygons near the shared
  region. This is for footprints that are deliberately conservative (arrays)
  but whose actual atoms don't collide.

`write_gds` calls `check()` and raises `ValueError` listing the problems,
unless `allow_overlap=True` (overlaps) or `min_spacing=None` (spacing not
checked).

### 5.4 Text

`text=` writes a `gdstk.Label` on `text_layer` at the footprint's bbox centre
(`text_anchor="center"`; also `"top"`, `"bottom"`, or explicit coordinates).
With `text_as_polygons=True` it writes `gdstk.text(...)` polygons instead,
still on `text_layer`, sized by `text_size`.

## 6. Phases

**G1 — tolerance-aware Shapely adapters**
- `tolerance=` through `shape_to_shapely` and every converter that draws a
  curve.
- Tests for each of the 12 primitives and for booleans:
  - every polygon vertex lies on the true outline (|sdf| ≤ tiny);
  - every edge midpoint is within `tolerance` of it (sdf ≥ −tolerance);
  - the vertex count drops when the tolerance is loosened.
- With `tolerance=None`, existing results don't change (the full suite stays
  green).

**G2 — conversion and cells**
- `gds/convert.py` and `gds/cells.py` as in §3.
- Tests:
  - Round trip: the area and Hausdorff distance between the written polygons
    (read back with `gdstk.read_gds`) and the Shapely geometry are within the
    tolerance and grid.
  - Holes, scaling with `coord_unit`, the vertex limit, and dropped
    sub-grid pieces.
  - The array cell contains one period plus a `Repetition` with the right
    v1/v2, including an oblique lattice.
  - Deduplication: references for repeated atoms, plain polygons for unique
    ones.
  - Clipping only touches crossing atoms.

**G3 — assembly**
- `GDSAssembly`, placements, `place_grid`, footprints, `check`, text,
  `write_gds`.
- Tests:
  - anchors (`center` / `origin`) and offsets;
  - footprint of an array whose atoms cross the cell boundary;
  - overlap and spacing detection, and touching allowed;
  - `write_gds` raises / `allow_overlap`;
  - a shared object becomes one cell;
  - labels only on the text layer;
  - the GDS round trip contains the expected cell tree.

**G4 — scale, docs, example**
- `benchmarks/gds_bench.py`:
  - 10⁴ / 10⁵ unique atoms;
  - a layout with 10⁵ atoms of 20 distinct types (deduplicated);
  - a 1000×1000 array.
- `workers=` only if the benchmark justifies it.
- `examples/gds.ipynb`: a chip with two aperiodic lenses and a grid of
  UnitCell batch arrays, a footprint plot, and `write_gds`.
- CLAUDE.md structure / key modules; §11-style implementation notes.

## 7. Out of scope (for now)

- A direct shape → gdstk path (only if G4 profiling demands it).
- Rotated or mirrored placements. gdstk supports them, and footprints would
  need rotating; easy to add later.
- Several layers per placement, or a layer per shape type.
- Merging overlapping atoms, and DRC beyond the placement-level checks.
- Reading GDS back into metashapes shapes.
- OASIS output. gdstk can write it with `write_oas`; add it if needed.

## 8. Implementation notes

### Deviations and details decided during implementation

| Topic | What was done |
|---|---|
| Tolerance plumbing | A `ContextVar` set by `shape_to_shapely(tolerance=)` / `tolerance_scope()`, read by `arc_quad_segs(radius, default)`, so the converters' signatures didn't change. `shape_to_shapely(tolerance=None)` **inherits** an enclosing scope (used for wrapped `Bar` cells via `UnitCell.to_shapely()`). `Scale` divides the tolerance by `|s|` for its child. |
| GEOS fillet rounding | GEOS splits a buffer fillet of angle A into `int(A/quantum + 0.5)` segments, so a partial arc's step can be 1.5 quanta. `arc_quad_segs(fillet=True)` (corner arcs from `buffer`) shrinks the quantum by 1.5. Full circles and parametric ellipses (`fillet=False`) don't need it. |
| Star outline | The opening-then-closing buffers were **ill-conditioned**. When the tip and valley fillets nearly meet on an edge (the sampler allows up to 99 %), the closing intersects almost-parallel offset curves, and the error grew like √tolerance (0.17 at tolerance 0.05). `star_to_shapely` now builds the outline directly (`_rounded_polygon`): exact tangent points, arcs sampled from their true centres, every vertex on the outline. Convex valleys (`r ≥ R cos(π/n)`) are rounded by `outer_corner_radius`, matching the SDF. |
| Infinite scenes in arrays | A finite scene is exported whole (decision 2). A scene with infinite extent (`Bar` gratings) can't be, so it is wrapped into the cell parallelogram and copies abut. Its footprint is the parallelogram. |
| Oblique arrays | gdstk writes an AREF only for axis-aligned, rectangular repetitions and expands anything else into one SREF per copy. `_array_references` therefore uses: **1 AREF** when a1/a2 lie along x/y; **one AREF per row** when only one vector is axis-aligned (n₂ references for a hexagonal lattice instead of n₁·n₂); gdstk's expansion only when neither is. Row origins are snapped to the database grid once, so there is no drift along a row. |
| Duplicate names | Rejected immediately by `place*` (`ValueError`), not reported by `check()`. Names are sanitised to `[A-Za-z0-9_?$]`. |
| Spacing semantics | With `min_spacing > 0`, touching footprints (distance 0) are a spacing violation. With `min_spacing=0`, touching is allowed and only shared area counts as overlap. |
| `place_grid` | Columns go +x and rows go −y from `origin` (the centre of the first slot). Items are centred by their footprint's bbox centre. Accepts `LayoutBatchResult.layouts` / `GenerationBatchResult.unit_cells` or a list. |
| Text | `gdstk.Label` at the footprint's bbox centre/top/bottom or an explicit point. `text_as_polygons` writes `gdstk.text` glyphs whose lower-left corner is at that point. |
| Deduplication | Bounds, `to_parametric()` and the key are memoised per distinct base object, since atoms made from one library shape share it. This took 10⁵ atoms of 20 kinds from 12 s to ≈1 s. `bounds()` runs before `to_parametric()`, so drifted parameters are clamped before they are keyed. |
| `workers` | `ProcessPoolExecutor` on a **`forkserver`** that preloads `metashapes.gds.cells`. Plain `fork` from a process running torch threads can deadlock the child: Python warns about it, and pytest showed the warning. The forkserver forks workers from a single-threaded server that already has the imports, so they start fast and safely. As with any non-fork start method, a *script* needs `if __name__ == "__main__":`; notebooks don't. Workers get `(to_parametric dict, dx, dy)` chunks and return WKB; gdstk objects are built in the main process. Used only for atoms that appear once, and only from 2·2000 such atoms. |

### Benchmark (`benchmarks/gds_bench.py`, 32-core CPU, model in nm, 1 nm tolerance)

| Case | Build | File |
|---|---|---|
| 10⁴ unique atoms (11 primitives) | 4.9 s (2.7 s with `workers=8`) | 2.5 MB |
| 10⁵ unique atoms | 50 s (20 s with `workers=8`) | 25 MB |
| 10⁵ atoms of 20 kinds (dedupe) | 1.0 s | 3.0 MB |
| 1000×1000 rectangular array | < 0.01 s | 0.0 MB (1 AREF) |
| 1000×1000 hexagonal array | < 0.01 s | 0.1 MB (1000 AREFs) |

Unique atoms cost ≈0.5 ms each, almost all of it Shapely conversion. With
workers, the serial rest is ≈0.2 ms per atom: `bounds()`, `to_parametric()`,
the dedupe key, WKB decoding and gdstk objects. So the speedup flattens at
about 2.5×. A direct shape → gdstk path for the common primitives (§7) is the
next step if this matters.
