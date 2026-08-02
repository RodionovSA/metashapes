# Graph Report - metashapes  (2026-08-01)

## Corpus Check
- 73 files · ~46,906 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1448 nodes · 4666 edges · 82 communities (54 shown, 28 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 1081 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8c210b3d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Shape Primitives Core|Shape Primitives Core]]
- [[_COMMUNITY_Random Generator & Lattice|Random Generator & Lattice]]
- [[_COMMUNITY_Shape Analysis & SDF Concepts|Shape Analysis & SDF Concepts]]
- [[_COMMUNITY_Unit Cell Analyzer|Unit Cell Analyzer]]
- [[_COMMUNITY_Conic Shape Primitives|Conic Shape Primitives]]
- [[_COMMUNITY_YAML Serialization Tests|YAML Serialization Tests]]
- [[_COMMUNITY_Shapely Transform Tests|Shapely Transform Tests]]
- [[_COMMUNITY_Random Generator Logic|Random Generator Logic]]
- [[_COMMUNITY_Shapely Compound Shape Tests|Shapely Compound Shape Tests]]
- [[_COMMUNITY_Generator Integration Tests|Generator Integration Tests]]
- [[_COMMUNITY_Generator Base Classes|Generator Base Classes]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Shapely Adapter Modules|Shapely Adapter Modules]]
- [[_COMMUNITY_YAML & Unit Cell Serialization|YAML & Unit Cell Serialization]]
- [[_COMMUNITY_Unit Cell Tests|Unit Cell Tests]]
- [[_COMMUNITY_Shapely Adapter Layer|Shapely Adapter Layer]]
- [[_COMMUNITY_PyTorch Differentiability|PyTorch Differentiability]]
- [[_COMMUNITY_Lattice SDF Tests|Lattice SDF Tests]]
- [[_COMMUNITY_Coordinate Transform Bridge|Coordinate Transform Bridge]]
- [[_COMMUNITY_Boolean Shape Tests|Boolean Shape Tests]]
- [[_COMMUNITY_Periodic Unit Cell Ops|Periodic Unit Cell Ops]]
- [[_COMMUNITY_Mask Tests|Mask Tests]]
- [[_COMMUNITY_Generator Validation|Generator Validation]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Test Package Root|Test Package Root]]
- [[_COMMUNITY_Shape Test Init|Shape Test Init]]
- [[_COMMUNITY_Lattice Test Init|Lattice Test Init]]
- [[_COMMUNITY_Adapters Test Init|Adapters Test Init]]
- [[_COMMUNITY_Generators Test Init|Generators Test Init]]
- [[_COMMUNITY_Lattice Basis Rationale|Lattice Basis Rationale]]
- [[_COMMUNITY_Lattice Basis Rationale B|Lattice Basis Rationale B]]
- [[_COMMUNITY_Lattice Basis Rationale C|Lattice Basis Rationale C]]
- [[_COMMUNITY_Lattice Package Init|Lattice Package Init]]
- [[_COMMUNITY_Validator Rationale|Validator Rationale]]
- [[_COMMUNITY_Generator Base Rationale|Generator Base Rationale]]
- [[_COMMUNITY_Analysis Test Suite|Analysis Test Suite]]
- [[_COMMUNITY_Primitives Init|Primitives Init]]
- [[_COMMUNITY_Sampler Utils|Sampler Utils]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 89|Community 89]]

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 155 edges
2. `Rectangle` - 147 edges
3. `Lattice` - 138 edges
4. `Ellipse` - 120 edges
5. `shape_to_shapely()` - 110 edges
6. `RegularPolygon` - 100 edges
7. `Bar` - 94 edges
8. `Shape` - 87 edges
9. `assert_inside()` - 76 edges
10. `Translate` - 70 edges

## Surprising Connections (you probably didn't know these)
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → src/metashapes/shape/primitives/conics.py
- `ConvexQuad` --semantically_similar_to--> `IsoscelesTrapezoid`  [INFERRED] [semantically similar]
  src/metashapes/shape/primitives/quads.py → metashapes/shape/primitives/quads.py
- `UnitCellAnalyzer.validate` --semantically_similar_to--> `UnitCellValidator`  [INFERRED] [semantically similar]
  metashapes/analysis.py → src/metashapes/generators/validator.py
- `TestCellMetrics` --uses--> `Union`  [INFERRED]
  tests/test_analysis.py → src/metashapes/shape/boolean.py
- `TestMetrics` --uses--> `Union`  [INFERRED]
  tests/test_analysis.py → src/metashapes/shape/boolean.py

## Communities (82 total, 28 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.11
Nodes (21): cell_area(), device(), dtype(), hexagonal(), matrix(), Coerce a constructor scalar to float, refusing grad-carrying tensors.      ``rec, rectangular(), _scalar() (+13 more)

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.06
Nodes (35): _centroid(), TestBooleansToShapely, TestCompoundShapesToShapely, TestPrimitivesToShapely, TestTransformsToShapely, Shapely Adapter Pattern, difference_to_shapely(), intersection_to_shapely() (+27 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.06
Nodes (49): TestBarSafeExtraction, TestCrossSafeExtraction, TestIsoscelesTrapezoidSafeExtraction, TestRegularPolygonSafeExtraction, TestStadiumSafeExtraction, TestStripeSafeExtraction, TestUnitCellSafeExtraction, CellMetrics (+41 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.05
Nodes (53): ABC, Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, register_shape_sampler(), SHAPE_SAMPLER_REGISTRY, Lattice, Cartesian translation for lattice cell (i, j)., Cartesian (x, y) -> fractional (f1, f2). (+45 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.12
Nodes (4): assert_gradients_finite_at(), Same as assert_gradients_finite but evaluates sdf() at a single,     possibly su, Same as assert_gradients_finite but evaluates sdf() at a single,     possibly su, TestEllipseDtypeDeviceGrad

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.19
Nodes (4): Rectangular unit cell with a small square at the origin., _square_cell(), TestExtent, TestUnitCellRasterize

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.12
Nodes (9): Ellipse, _ellipse_closest_point(), Ellipse.min_feature_size, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Nearest point on the boundary of an axis-aligned ellipse (semi-axes     a, b, ce (+1 more)

### Community 7 - "Random Generator Logic"
Cohesion: 0.11
Nodes (11): UnitCellAnalyzer.validate, _gen(), test_each_shape_type_generates(), TestBasicGeneration, TestConstraints, TestReport, TestShapeCount, TestShapeTypes (+3 more)

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.16
Nodes (13): GeneratorConfig, RandomUnitCellGenerator._generate_one, RandomGeneratorConfig, RandomUnitCellGenerator, First concrete generator config.      For now it only adds one flag:     - requi, First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, Random unit-cell generator.      Current responsibilities:     1. choose number (+5 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.07
Nodes (18): Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, assert_inside(), assert_round_trip() (+10 more)

### Community 10 - "Generator Base Classes"
Cohesion: 0.12
Nodes (6): General triangle defined by two base angles and the base length (ASA).      Para, (A, B, C) as (x, y) tensor pairs, CCW, centroid at origin., Triangle, assert_outside(), All points should have SDF > -tol (outside or on boundary)., TestTriangle

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (9): _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator, GeneratorConfig, _has_infinite_bounds() (+1 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.09
Nodes (4): Bar, An infinite bar spanning the full unit cell along one axis.      The bar is unbo, TestBar, TestBarDtypeDeviceGrad

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.29
Nodes (7): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, load_unit_cells(), Save one or more unit cells to a YAML file.      Parameters     ----------     p, Load unit cells from a YAML file saved by :func:`save_unit_cells`     or :func:`, save_unit_cells()

### Community 14 - "Unit Cell Tests"
Cohesion: 0.10
Nodes (6): square_in_rect(), Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, RegularPolygon, TestRegularPolygon, TestRegularPolygonDtypeDeviceGrad

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.13
Nodes (7): IsoscelesTrapezoid, Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Shape, TestIsoscelesTrapezoid, TestIsoscelesTrapezoidDtypeDeviceGrad

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.33
Nodes (6): _brute_force_sdf(), Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, test_matches_brute_force()

### Community 17 - "Lattice SDF Tests"
Cohesion: 0.09
Nodes (5): ConvexQuad, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, TestConvexQuad, TestConvexQuadDtypeDeviceGrad

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.11
Nodes (6): Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Star, TestStar, TestStarDtypeDeviceGrad

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.15
Nodes (18): _make_batch_result(), TestSaveBatchResult, _check_version(), _Dumper, load_batch_result(), _load_cell(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Recursively convert tuples → lists so yaml.dump produces clean YAML. (+10 more)

### Community 20 - "Periodic Unit Cell Ops"
Cohesion: 0.11
Nodes (16): _decompose_for_offsets(), Periodic signed distance of the scene at Cartesian (x, y).          Query points, Return world-coordinate points on the material boundary (zero-level-set)., A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, Yield sub-shapes to measure separately when sizing the periodic     copy search,, Return world-coordinate points on the material boundary (zero-level-set)., Return world-coordinate points on the material boundary (zero-level-set)., Shapely geometry of the scene, clipped to the unit cell.          Periodic, matc (+8 more)

### Community 21 - "Mask Tests"
Cohesion: 0.12
Nodes (16): Analysis, Boolean Operations, code:bash (pip install git+https://github.com/RodionovSA/Metashapes), code:python (from metashapes import UnitCell, Lattice), code:python (from metashapes.shape import Ellipse, Rectangle), code:python (from metashapes import Lattice), code:python (from metashapes import UnitCellAnalyzer), code:python (from metashapes.adapters.yaml import save_unit_cells, load_u) (+8 more)

### Community 22 - "Generator Validation"
Cohesion: 0.16
Nodes (15): gdstk_to_shapely(), klayout_to_shapely(), numpy_to_shapely(), Convert a shapely geometry to a gdstk polygon or a list of polygons., Convert a gdstk polygon to a shapely geometry., Convert a shapely geometry to a KLayout polygon or a list of polygons., Create a shapely geometry from a binary numpy array.     Parameters:         img, shapely_to_gdstk() (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (15): code:python (a = torch.tensor([1.0], requires_grad=True)), code:block11 (fractional_grid(4,4) row:              [0.0, 0.25, 0.5, 0.75), code:python (from metashapes.lattice import Lattice   # ImportError), L-04 — Rasterize's two sampling paths use different conventions for the same `nx`/`ny`, L-05 — `lattice/__init__.py` is empty; `Lattice` isn't importable from the package, Medium, S-07 — `torch.empty_like` + `torch.where` is a latent NaN-gradient hazard, S-08 — Quilez ellipse-SDF solver duplicated ~100 lines between `Ellipse` and `Egg` (+7 more)

### Community 24 - "Community 24"
Cohesion: 0.10
Nodes (16): _cell_center(), Rectangular cell with a Rectangle shape centred at (cx, cy)., For a symmetric Rectangle, both methods produce the same offset., For a centred Ellipse, both methods produce the same offset., The geometric midpoint (a1 + a2) / 2 as plain floats., Extract the outermost Translate dx/dy buffers from the scene., _rect_cell(), TestAlreadyCentered (+8 more)

### Community 25 - "Community 25"
Cohesion: 0.07
Nodes (16): Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, is_empty_bounds(), _rect(), TestDifferenceBounds, TestIntersectionBounds, TestIsEmptyBounds (+8 more)

### Community 26 - "Test Package Root"
Cohesion: 0.15
Nodes (12): code:bash (graphify query "<your question>"       # any codebase questi), code:bash (source .venv/bin/activate), code:block3 (src/metashapes/          # src-layout; imported as `metashap), graphify, Key Modules, Metashapes — Developer Guide, Navigating This Codebase, New Shape Primitive Addition (+4 more)

### Community 27 - "Shape Test Init"
Cohesion: 0.40
Nodes (5): sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, sdf() must agree with a large brute-force reference search for a     variety of, sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, sdf() must agree with a large brute-force reference search for a     variety of, TestPeriodicSdfMatchesBruteForce

### Community 28 - "Lattice Test Init"
Cohesion: 0.11
Nodes (11): assert_direct_call_dtype_promotion(), assert_dtype_device_flow(), assert_gradients_finite(), _assert_param_grads_finite(), Calling sdf() directly on a primitive (bypassing UnitCell) with     float64 quer, Calling sdf() directly on a primitive (bypassing UnitCell) with     float64 quer, Grid-evaluate sdf() over `x_range`, sum, backward(); every named     parameter i, Grid-evaluate sdf() over `x_range`, sum, backward(); every named     parameter i (+3 more)

### Community 29 - "Adapters Test Init"
Cohesion: 0.12
Nodes (6): TestOffsetSearchOverhead, TestOffsetsForRing, TestToShapely, Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Rectangle, TestRectangleDtypeDeviceGrad

### Community 31 - "Lattice Basis Rationale"
Cohesion: 0.15
Nodes (11): nn.Module Subclassing for Differentiability, make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution., Gradient flows from a point displaced by one lattice vector., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution. (+3 more)

### Community 32 - "Lattice Basis Rationale B"
Cohesion: 0.11
Nodes (6): Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium, TestStadium, TestStadiumDtypeDeviceGrad

### Community 35 - "Validator Rationale"
Cohesion: 0.12
Nodes (9): SDF Convention: Negative Inside, Positive Outside, Egg, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, sdf_at(), test_matches_brute_force_nearest_point(), TestEgg (+1 more)

### Community 36 - "Generator Base Rationale"
Cohesion: 0.20
Nodes (3): assert_bounds_contain(), All points should lie inside (or on) the reported bounding box., TestRectangle

### Community 52 - "Community 52"
Cohesion: 0.10
Nodes (11): Parametric Serialization (to_parametric / from_parametric), Base class for all symbolic 2D shapes., Signed distance evaluated on torch tensors.         x, y can be broadcastable te, True if `bounds` (as returned by `Shape.bounds()`) is an empty/inverted box., Base class for all symbolic 2D shapes., Signed distance evaluated on torch tensors.         x, y can be broadcastable te, Alias for `sdf`, so a Shape can be called directly (`shape(x, y)`)         follo, Axis-aligned bounding box of the shape, in world coordinates.          Returns ( (+3 more)

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (6): Direct tests of the shared helper, plus a check that Rectangle's own     sdf() a, Direct tests of the shared helper, plus a check that Rectangle's own     sdf() a, TestSdfRoundedBox, SDF of an axis-aligned box of half-extents (hx, hy) with corners     rounded by, _sdf_rounded_box(), _to_local_coords()

### Community 55 - "Community 55"
Cohesion: 0.19
Nodes (6): from_parametric(), min_feature_size(), Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, to_plain_data()

### Community 65 - "Community 65"
Cohesion: 0.12
Nodes (9): p(), pv(), center_scene(method='centroid') must work even if params have grad., Verifies that calling shape_to_shapely (which does .detach().cpu())         does, Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestEllipseSafeExtraction, TestRectangleSafeExtraction (+1 more)

### Community 68 - "Community 68"
Cohesion: 0.18
Nodes (10): code:block5 (Rectangle(size=[0.3, 0.8]).min_feature_size          -> 0.3), code:python (Rotate.from_parametric({"type": "Rotate", "shape": ..., "ang), code:block8 (f1 = [-inf, -inf, inf, inf]), code:block9 (_ring_for -> (2, 2)  =>  25 full scene.sdf() evaluations per), High, L-02 — `_ring_for` correctness currently depends on `0 × inf → NaN`, L-03 — `UnitCell.sdf` costs ~22× a single shape evaluation for small shapes, S-04 — `min_feature_size` is silently lost through every transform and boolean (+2 more)

### Community 70 - "Community 70"
Cohesion: 0.20
Nodes (13): Smooth Boolean Operations via Polynomial Blending, Difference, from_parametric(), Intersection, Symbolic difference of two shapes: left - right., Symbolic difference of two shapes: left - right., Symbolic difference of two shapes: left - right.      Note: like `Intersection`,, Symbolic union of two shapes. (+5 more)

### Community 72 - "Community 72"
Cohesion: 0.22
Nodes (9): code:block1 (Egg(center=[0,0], width=2.0, height=2.5, skew=0.6)   # a=1, ), code:python (lat = Lattice.rectangular(1.0, 1.0)), code:block3 (rr=0.9: constructed OK, sdf OK), code:block4 (y=0.0000: sdf=+0.00000        (center — should be strongly n), Critical, L-01 — `UnitCell.to_shapely()` is non-periodic while `UnitCell.sdf()` is periodic, S-01 — `Egg.sdf` is discontinuous and sign-wrong across the y=0 seam, S-02 — `ConvexQuad` corner-radius validity check is non-monotonic and fails open (+1 more)

### Community 73 - "Community 73"
Cohesion: 0.22
Nodes (9): L-06 — Stale docstring contradicts the code, L-07 — `fractional_grid` is unreachable public API, Low, S-16 — Unused `BaseGeometry` import in `shape/base.py`, S-17 — `Shape(nn.Module)` never defines `forward()`, S-18 — `Star.sdf` recomputes `sin_beta` and shadows loop variables, S-19 — `outer_corner_radius`/`inner_corner_radius` deviate from the `corner_radius` convention, undocumented as an exception, S-20 — Inconsistent bounds tightness with no stated contract (+1 more)

### Community 76 - "Community 76"
Cohesion: 0.40
Nodes (3): nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., TestGradientFlow

### Community 78 - "Community 78"
Cohesion: 0.13
Nodes (12): min_feature_size(), min_feature_size(), min_feature_size(), _max_corner_radius(), min_feature_size(), _quad_vertices(), _signed_area2(), register_shape() (+4 more)

### Community 84 - "Community 84"
Cohesion: 0.40
Nodes (5): Post-screening findings (from the dtype/device/gradient test pass), S-22 — `IsoscelesTrapezoid` never got the S-02 treatment ✅ Fixed, S-23 — `Ellipse`/`Egg`'s cubic-solve has a real cube-root gradient singularity ✅ Fixed, S-24 — Branch-1 `acos(±1)` gradient singularity on any on-axis query ✅ Fixed, S-25 — Near-circular ellipses overflow `c³` in float32, producing NaN *values* ✅ Fixed

### Community 85 - "Community 85"
Cohesion: 0.40
Nodes (4): Screening: `src/metashapes/shape/` + `src/metashapes/lattice/`, Severity / Priority / Effort, Suggested fix order (P0 → P3), Summary

### Community 89 - "Community 89"
Cohesion: 0.50
Nodes (4): Inverse Design for Metasurfaces, SDF-based differentiable shapes concept, Shape + Lattice = UnitCell model, MetaShapes Project Documentation

## Knowledge Gaps
- **61 isolated node(s):** `allow`, `PreToolUse`, `code:bash (graphify query "<your question>"       # any codebase questi)`, `Overview`, `code:bash (source .venv/bin/activate)` (+56 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Shape` connect `Community 52` to `Shape Primitives Core`, `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `Shapely Transform Tests`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Generator Base Classes`, `Community 11`, `Shapely Adapter Modules`, `Unit Cell Tests`, `Shapely Adapter Layer`, `Lattice SDF Tests`, `Coordinate Transform Bridge`, `Lattice Test Init`, `Adapters Test Init`, `Lattice Basis Rationale`, `Lattice Basis Rationale B`, `Validator Rationale`, `Community 55`, `Community 70`, `Community 78`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `Rectangle` connect `Adapters Test Init` to `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Shapely Adapter Modules`, `YAML & Unit Cell Serialization`, `Shapely Adapter Layer`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Boolean Shape Tests`, `Community 24`, `Community 25`, `Shape Test Init`, `Lattice Basis Rationale`, `Generator Base Rationale`, `Community 52`, `Community 53`, `Community 61`, `Community 65`, `Community 71`, `Community 74`, `Community 75`, `Community 76`, `Community 78`, `Community 80`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `UnitCell` connect `Periodic Unit Cell Ops` to `Shape Primitives Core`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Community 11`, `YAML & Unit Cell Serialization`, `Unit Cell Tests`, `PyTorch Differentiability`, `Boolean Shape Tests`, `Community 24`, `Community 25`, `Shape Test Init`, `Adapters Test Init`, `Lattice Basis Rationale`, `Community 61`, `Community 65`, `Community 71`, `Community 74`, `Community 75`, `Community 76`, `Community 80`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 84 inferred relationships involving `UnitCell` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`UnitCell` has 84 INFERRED edges - model-reasoned connections that need verification._
- **Are the 101 inferred relationships involving `Rectangle` (e.g. with `Shape` and `TestLeafShapes`) actually correct?**
  _`Rectangle` has 101 INFERRED edges - model-reasoned connections that need verification._
- **Are the 86 inferred relationships involving `Lattice` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`Lattice` has 86 INFERRED edges - model-reasoned connections that need verification._
- **Are the 67 inferred relationships involving `Ellipse` (e.g. with `Shape` and `TestLeafShapes`) actually correct?**
  _`Ellipse` has 67 INFERRED edges - model-reasoned connections that need verification._