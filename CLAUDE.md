# Metashapes — Developer Guide

## Navigating This Codebase

**Before reading files or grepping:** check the knowledge graph first.

```bash
graphify query "<your question>"       # any codebase question
graphify path "<A>" "<B>"              # how two things relate
graphify explain "<concept>"           # focused concept breakdown
```

If `graphify-out/graph.json` exists, these commands return scoped subgraphs faster than raw search. Fall back to file reads only when graphify output is insufficient.

---

## Overview

Python library for 2D shape management in metasurface design workflows. Core idea: shapes are defined analytically via **Signed Distance Functions (SDF)**, making them differentiable (PyTorch-backed) and composable via boolean operations. Shapes live in a `UnitCell` with a `Lattice` defining periodicity.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/
```
> Uses uv for environment management. See pyproject.toml for setup.


## Repository Structure

```
src/metashapes/          # src-layout; imported as `metashapes`
  shape/
    base.py          # Shape base class (SDF, bounds, boolean operators)
    boolean.py       # Union, Intersection, Difference
    transforms.py    # Translate, Rotate, Scale
    shapeset.py      # ShapeSet — flat, spatially indexed union of many atoms (layout scenes)
    spatial.py       # SpatialHash — torch-side uniform grid index used by ShapeSet
    learnable.py     # make_learnable() — post-hoc buffer -> nn.Parameter conversion
    registry.py      # register_shape() — name registry for serialization
    utils.py         # register() param helper, _to_local_coords()
    primitives/
      quads.py       # Rectangle, ConvexQuad, IsoscelesTrapezoid
      ovals.py       # Ellipse, Egg, Stadium
      polygons.py    # RegularPolygon, Triangle, Star
      junctions.py   # Cross, TShape
      stripes.py     # Bar — 1D-infinite, spans the cell along one axis

  lattice/
    basis.py         # Lattice — 2D periodic basis vectors
    unit_cell.py     # UnitCell — lattice + shape, periodic SDF sampling
    grid.py          # fractional_grid(), cartesian_grid()

  generators/
    base.py          # UnitCellGenerator ABC
    random.py        # RandomUnitCellGenerator
    config.py        # GeneratorConfig dataclass
    validator.py     # UnitCellValidator (gap/feature size checks)
    report.py        # GenerationBatchResult, GenerationReport
    registry.py      # register_shape_sampler() — sampler registry
    samplers/        # One file per primitive family
      base.py        # ShapeSampler ABC
      utils.py       # lattice_inner_bounds(), resolve_param()

  adapters/
    shapely/
      dispatch.py    # shape_to_shapely() — isinstance dispatch entry point
      quads.py ovals.py polygons.py junctions.py stripes.py   # per-primitive converters
      booleans.py transforms.py    # recursive converters for composite nodes
      helpers.py     # as_float(), round_corners(), remove_holes()
      raster.py      # shapely_to_numpy()
    yaml.py          # YAML serialization / deserialization (unit cells and layouts)
    npz.py           # Columnar .npz save/load for large layouts

  layout/            # Non-periodic layouts (see docs/layout_design.md); nothing else imports it
    layout.py        # Layout — domain + ShapeSet, rasterize/mask/tiles, checks, containment_penalty
    sites.py         # LatticeSites, RandomSites, ExplicitSites — where candidates go
    sources.py       # SamplerSource, ParametricSource, Regions — what atom goes there
    config.py        # LayoutConfig
    generator.py     # LayoutGenerator, LayoutBatchResult, LayoutValidator
    spatial.py       # GeometryHash — incremental Shapely index for generation
    analysis.py      # LayoutAnalyzer, LayoutMetrics

  analysis.py        # UnitCellAnalyzer — fill factor, gap, feature size metrics
  transform.py       # Interop: Shapely ↔ NumPy ↔ gdstk ↔ KLayout

tests/
  shape/             # Per-primitive shape tests
  lattice/           # Basis and UnitCell tests
  adapters/          # Shapely + YAML adapter tests
  generators/        # Generator tests
  layout/            # Layout rendering, generation, analysis tests
  test_analysis.py   # UnitCellAnalyzer tests
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `shape/base.py` | `Shape` — all primitives subclass this; defines `sdf()`, `bounds()`, boolean operators |
| `lattice/unit_cell.py` | `UnitCell` — composes a `Lattice` and a `Shape` into a periodic structure |
| `lattice/basis.py` | `Lattice` — holds `a1`, `a2` basis vectors; handles fractional↔Cartesian coords |
| `generators/random.py` | `RandomUnitCellGenerator` — constraint-based random cell generation |
| `generators/config.py` | `GeneratorConfig` — target count, allowed shapes, min gap/feature, seed |
| `generators/validator.py` | `UnitCellValidator` — rejects cells that violate geometric constraints |
| `analysis.py` | `UnitCellAnalyzer` — fill factor, min gap, min feature size via Shapely |
| `adapters/shapely/` | Converts any `Shape` tree to `shapely.Polygon` for analysis and export |
| `adapters/shapely/dispatch.py` | `shape_to_shapely()` — isinstance dispatch entry point every adapter question routes through |
| `adapters/yaml.py` | Save/load lists of `UnitCell` / `Layout` to YAML |
| `shape/shapeset.py` | `ShapeSet` — atoms evaluated via spatial index; per-call `halo` (default `inf` = exact) |
| `layout/layout.py` | `Layout` — non-periodic counterpart of `UnitCell`; renders `scene ∩ domain` |
| `layout/generator.py` | `LayoutGenerator` — sites × sources × constraints (containment, gap, feature) |
| `shape/registry.py` | `register_shape()` — name registry for serialization |
| `generators/registry.py` | `register_shape_sampler()` — sampler registry |

## Notes

- Several files exist as placeholders or are kept for future use — treat unfamiliar files carefully before modifying.
- SDF convention: negative inside the shape, positive outside.
- Shapes are `nn.Module` subclasses so parameters are differentiable by default.
- All 12 primitives have both a Shapely converter (`dispatch.py`) and a sampler. Samplers register
  via `@register_shape_sampler` when their module is imported, so `SHAPE_SAMPLER_REGISTRY` — not
  `__all__` in `generators/samplers/__init__.py` — is the source of truth for what
  `RandomUnitCellGenerator` can emit.
- `min_feature_size` uses **full** dimensions (e.g. Ellipse: min full axis, Egg: `min(width, height)`,
  Stadium: full width), matching how samplers compare against `config.min_feature_size`. Don't
  return half-axes/radii — samplers would then emit shapes the generator rejects.

## New Shape Primitive Addition

New shape primitives are added regularly. Follow this process every time.

**Before starting — ask the user if any of these are missing:**
- Full description of the shape and all its geometric parameters.
- Which primitives file it belongs to (e.g. `quads.py`, `ovals.py`) or whether a new file is needed.

**Design constraint:**
- Shapes with corners **must** include a `corner_radius` parameter for optional corner smoothing.
  Exception: shapes with both convex and concave corners needing two independent radii (e.g.
  `Cross`, `TShape`, `Star`) use `outer_corner_radius`/`inner_corner_radius` instead — this is an
  intentional, documented deviation from the single-`corner_radius` convention, not an oversight.
- `sdf()` must be differentiable and free of data-dependent Python control flow: no `if tensor:`,
  no `.item()` used to decide which branch runs, no data-dependent Python loop over a per-point
  condition. Express branches with `torch.where`/masking so both sides are evaluated as ordinary
  tensor ops — this keeps `sdf()` traceable (`torch.compile`, `vmap`) and avoids a `.item()`
  forcing a GPU→CPU sync on every call. Exception: an `.item()`/`if tensor:` used only to `raise`
  a clear construction-time-unreachable validation error (not to select a different SDF value) is
  fine — replacing a clear exception with silent NaN production to satisfy this rule would be a
  worse outcome, not a better one.

**Known, deliberate exclusions** (not oversights — don't re-propose these without new information):
- Shape parameters are **not** re-validated after construction via a raised error — but most
  primitives now implement `_project()` (called at the top of `sdf()`/`bounds()`/
  `min_feature_size`) to clamp drifted parameters back into their valid range in place, so an
  optimizer step doesn't need to keep them there itself. Every primitive in `quads.py`,
  `stripes.py`, `junctions.py`, and `polygons.py` has one; `ovals.py` (`Ellipse`, `Egg`,
  `Stadium`) is the current gap.
- Dtype is **not** threaded through constructors — everything defaults to float32 at construction
  time, same as before. Getting a different dtype (e.g. float64) is the caller's job via
  `.to(dtype=...)` on the constructed `Shape`/`Lattice`/`UnitCell`, the same pattern already used
  for device placement.

**Implementation checklist (all four are required):**
1. Shape class → `src/metashapes/shape/primitives/<file>.py`
2. Shapely adapter → `src/metashapes/adapters/shapely/<file>.py`
3. Parameter sampler → `src/metashapes/generators/samplers/<file>.py`
4. Tests → `tests/shape/`, `tests/adapters/`, `tests/generators/`

**Validation:** Run `pytest tests/` and confirm all new tests pass before reporting done.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
