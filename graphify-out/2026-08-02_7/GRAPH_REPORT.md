# Graph Report - metashapes  (2026-08-02)

## Corpus Check
- 76 files · ~47,225 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1596 nodes · 5110 edges · 89 communities (58 shown, 31 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 1287 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c3991bd9`
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
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 155 edges
2. `Rectangle` - 149 edges
3. `Lattice` - 141 edges
4. `Ellipse` - 120 edges
5. `shape_to_shapely()` - 113 edges
6. `RegularPolygon` - 100 edges
7. `Bar` - 94 edges
8. `Shape` - 90 edges
9. `Ellipse` - 90 edges
10. `assert_inside()` - 87 edges

## Surprising Connections (you probably didn't know these)
- `UnitCellAnalyzer.validate` --semantically_similar_to--> `UnitCellValidator`  [INFERRED] [semantically similar]
  metashapes/analysis.py → src/metashapes/generators/validator.py
- `ConvexQuad` --semantically_similar_to--> `IsoscelesTrapezoid`  [INFERRED] [semantically similar]
  src/metashapes/shape/primitives/quads.py → metashapes/shape/primitives/quads.py
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → src/metashapes/shape/primitives/conics.py
- `TestLeafShapes` --uses--> `CellMetrics`  [INFERRED]
  tests/test_analysis.py → src/metashapes/analysis.py
- `TestMetrics` --uses--> `CellMetrics`  [INFERRED]
  tests/test_analysis.py → src/metashapes/analysis.py

## Communities (89 total, 31 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.06
Nodes (34): cell_area(), device(), dtype(), hexagonal(), matrix(), Coerce a constructor scalar to float, refusing grad-carrying tensors.      ``rec, rectangular(), _scalar() (+26 more)

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.12
Nodes (10): Parametric Serialization (to_parametric / from_parametric), Base class for all symbolic 2D shapes., Signed distance evaluated on torch tensors.         x, y can be broadcastable te, True if `bounds` (as returned by `Shape.bounds()`) is an empty/inverted box., Base class for all symbolic 2D shapes., Signed distance evaluated on torch tensors.         x, y can be broadcastable te, Alias for `sdf`, so a Shape can be called directly (`shape(x, y)`)         follo, Axis-aligned bounding box of the shape, in world coordinates.          Returns ( (+2 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.09
Nodes (23): _compute_min_gap, _leaf_shapes, UnitCellAnalyzer, UnitCellAnalyzer.validate, _compute_min_gap(), _leaf_shapes(), Compute all metrics for a single cell., Return a list of constraint violation descriptions.         An empty list means (+15 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.05
Nodes (52): register_shape_sampler(), Lattice, Cartesian translation for lattice cell (i, j)., Cartesian (x, y) -> fractional (f1, f2)., Fractional (f1, f2) -> Cartesian (x, y)., Cartesian translation for lattice cell (i, j)., In-plane periodicity of the unit cell.     Defined by two lattice vectors. A rec, In-plane periodicity of the unit cell. Fixed (non-optimizable).     Defined by t (+44 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.14
Nodes (8): SDF Convention: Negative Inside, Positive Outside, assert_outside(), All points should have SDF < tol (inside or on boundary)., All points should have SDF > -tol (outside or on boundary)., sdf_at(), test_matches_brute_force_nearest_point(), TestEllipse, test_matches_brute_force_nearest_point()

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.09
Nodes (8): Rectangular unit cell with a small square at the origin., _sdf_at(), _square_cell(), TestExtent, TestUnitCellBoundary, TestUnitCellMask, TestUnitCellRasterize, TestUnitCellSDF

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.18
Nodes (11): TestStripeSafeExtraction, TestCompoundShapesToShapely, TestTransformsToShapely, Symbolic scaling of a shape., Symbolic scaling of a shape., Symbolic translation of a shape., Symbolic rotation of a shape., Symbolic rotation of a shape. (+3 more)

### Community 7 - "Random Generator Logic"
Cohesion: 0.11
Nodes (6): _gen(), TestBasicGeneration, TestConstraints, TestReport, TestShapeCount, TestShapeTypes

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.16
Nodes (13): GeneratorConfig, RandomUnitCellGenerator._generate_one, RandomGeneratorConfig, RandomUnitCellGenerator, First concrete generator config.      For now it only adds one flag:     - requi, First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, Random unit-cell generator.      Current responsibilities:     1. choose number (+5 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.05
Nodes (24): Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, Shape, assert_bounds_contain() (+16 more)

### Community 10 - "Generator Base Classes"
Cohesion: 0.09
Nodes (5): General triangle defined by two base angles and the base length (ASA).      Para, (A, B, C) as (x, y) tensor pairs, CCW, centroid at origin., Triangle, TestTriangle, TestTriangleDtypeDeviceGrad

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (17): ABC, _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator, GeneratorConfig (+9 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.09
Nodes (4): Bar, An infinite bar spanning the full unit cell along one axis.      The bar is unbo, TestBar, TestBarDtypeDeviceGrad

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.18
Nodes (13): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, _check_version(), _Dumper, _load_cell(), load_unit_cells(), Recursively convert tuples → lists so yaml.dump produces clean YAML. (+5 more)

### Community 14 - "Unit Cell Tests"
Cohesion: 0.11
Nodes (6): Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, RegularPolygon, assert_inside(), TestRegularPolygon, TestRegularPolygonDtypeDeviceGrad

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.08
Nodes (15): IsoscelesTrapezoid, Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), ConvexQuadSDF() (+7 more)

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.40
Nodes (5): sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, sdf() must agree with a large brute-force reference search for a     variety of, sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, sdf() must agree with a large brute-force reference search for a     variety of, TestPeriodicSdfMatchesBruteForce

### Community 17 - "Lattice SDF Tests"
Cohesion: 0.10
Nodes (9): ConvexQuad, _max_corner_radius(), _quad_vertices(), Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, _signed_area2() (+1 more)

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.10
Nodes (6): Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Star, TestStar, TestStarDtypeDeviceGrad

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.23
Nodes (9): _make_batch_result(), TestSaveBatchResult, load_batch_result(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Save a :class:`~metashapes.generators.report.GenerationBatchResult`     (unit ce, save_batch_result(), _sdf_grid(), test_each_shape_type_generates() (+1 more)

### Community 20 - "Periodic Unit Cell Ops"
Cohesion: 0.09
Nodes (6): Egg, _ellipse_closest_point(), Nearest point on the boundary of an axis-aligned ellipse (semi-axes     a, b, ce, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, TestEgg, TestEggDtypeDeviceGrad

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
Cohesion: 0.14
Nodes (7): _disjoint_intersection(), Union/Intersection/Difference deliberately return None: combining     shapes can, _rect(), TestEmptyBoundsThroughTransforms, TestFromParametricOriginDefault, TestMinFeatureSizeBooleansStayUnknown, TestMinFeatureSizePropagation

### Community 26 - "Test Package Root"
Cohesion: 0.15
Nodes (12): code:bash (graphify query "<your question>"       # any codebase questi), code:bash (source .venv/bin/activate), code:block3 (src/metashapes/          # src-layout; imported as `metashap), graphify, Key Modules, Metashapes — Developer Guide, Navigating This Codebase, New Shape Primitive Addition (+4 more)

### Community 27 - "Shape Test Init"
Cohesion: 0.19
Nodes (6): is_empty_bounds(), TestIsEmptyBounds, bounded(), positive(), Make parameter stricly positive, Make parameter stricly bounded

### Community 28 - "Lattice Test Init"
Cohesion: 0.09
Nodes (11): assert_direct_call_dtype_promotion(), assert_dtype_device_flow(), assert_gradients_finite(), _assert_param_grads_finite(), Calling sdf() directly on a primitive (bypassing UnitCell) with     float64 quer, Calling sdf() directly on a primitive (bypassing UnitCell) with     float64 quer, Grid-evaluate sdf() over `x_range`, sum, backward(); every named     parameter i, Grid-evaluate sdf() over `x_range`, sum, backward(); every named     parameter i (+3 more)

### Community 29 - "Adapters Test Init"
Cohesion: 0.15
Nodes (11): min_feature_size(), min_feature_size(), min_feature_size(), min_feature_size(), _project(), register_shape(), Register `value` on `module` under `name`.      If `value` is an nn.Parameter it, Register `value` on `module` under `name`.      If `value` is an nn.Parameter it (+3 more)

### Community 31 - "Lattice Basis Rationale"
Cohesion: 0.11
Nodes (14): nn.Module Subclassing for Differentiability, make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., TestGradientFlow, Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution. (+6 more)

### Community 32 - "Lattice Basis Rationale B"
Cohesion: 0.11
Nodes (6): Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium, TestStadium, TestStadiumDtypeDeviceGrad

### Community 35 - "Validator Rationale"
Cohesion: 0.13
Nodes (6): Egg, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, TestEgg, TestEggDtypeDeviceGrad

### Community 36 - "Generator Base Rationale"
Cohesion: 0.14
Nodes (4): Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium, TestStadium, TestStadiumDtypeDeviceGrad

### Community 52 - "Community 52"
Cohesion: 0.12
Nodes (20): Smooth Boolean Operations via Polynomial Blending, from_parametric(), min_feature_size(), Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, to_plain_data(), Difference (+12 more)

### Community 53 - "Community 53"
Cohesion: 0.11
Nodes (15): p(), pv(), center_scene(method='centroid') must work even if params have grad., Verifies that calling shape_to_shapely (which does .detach().cpu())         does, Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestBarSafeExtraction, TestCrossSafeExtraction (+7 more)

### Community 55 - "Community 55"
Cohesion: 0.16
Nodes (11): CellMetrics, Rotation Guard for Infinite-Extent Shapes, _bbox_size(), CellMetrics, True if this periodic shift should be skipped for self-gap measurement.      A s, Per-cell measurements produced by :class:`UnitCellAnalyzer`.      Attributes, _skip_self_shift(), _to_geom() (+3 more)

### Community 61 - "Community 61"
Cohesion: 0.12
Nodes (5): Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Rectangle, TestRectangle, TestRectangleDtypeDeviceGrad

### Community 65 - "Community 65"
Cohesion: 0.27
Nodes (4): _rect(), TestDifferenceBounds, TestIntersectionBounds, TestUnionBoundsWithEmptyChild

### Community 68 - "Community 68"
Cohesion: 0.18
Nodes (10): code:block5 (Rectangle(size=[0.3, 0.8]).min_feature_size          -> 0.3), code:python (Rotate.from_parametric({"type": "Rotate", "shape": ..., "ang), code:block8 (f1 = [-inf, -inf, inf, inf]), code:block9 (_ring_for -> (2, 2)  =>  25 full scene.sdf() evaluations per), High, L-02 — `_ring_for` correctness currently depends on `0 × inf → NaN`, L-03 — `UnitCell.sdf` costs ~22× a single shape evaluation for small shapes, S-04 — `min_feature_size` is silently lost through every transform and boolean (+2 more)

### Community 69 - "Community 69"
Cohesion: 0.11
Nodes (7): Ellipse, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, assert_gradients_finite_at(), Same as assert_gradients_finite but evaluates sdf() at a single,     possibly su, Same as assert_gradients_finite but evaluates sdf() at a single,     possibly su, TestEllipse, TestEllipseDtypeDeviceGrad

### Community 71 - "Community 71"
Cohesion: 0.09
Nodes (13): Ellipse, _ellipse_closest_point(), Ellipse.min_feature_size, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Nearest point on the boundary of an axis-aligned ellipse (semi-axes     a, b, ce (+5 more)

### Community 72 - "Community 72"
Cohesion: 0.22
Nodes (9): code:block1 (Egg(center=[0,0], width=2.0, height=2.5, skew=0.6)   # a=1, ), code:python (lat = Lattice.rectangular(1.0, 1.0)), code:block3 (rr=0.9: constructed OK, sdf OK), code:block4 (y=0.0000: sdf=+0.00000        (center — should be strongly n), Critical, L-01 — `UnitCell.to_shapely()` is non-periodic while `UnitCell.sdf()` is periodic, S-01 — `Egg.sdf` is discontinuous and sign-wrong across the y=0 seam, S-02 — `ConvexQuad` corner-radius validity check is non-monotonic and fails open (+1 more)

### Community 73 - "Community 73"
Cohesion: 0.22
Nodes (9): L-06 — Stale docstring contradicts the code, L-07 — `fractional_grid` is unreachable public API, Low, S-16 — Unused `BaseGeometry` import in `shape/base.py`, S-17 — `Shape(nn.Module)` never defines `forward()`, S-18 — `Star.sdf` recomputes `sin_beta` and shadows loop variables, S-19 — `outer_corner_radius`/`inner_corner_radius` deviate from the `corner_radius` convention, undocumented as an exception, S-20 — Inconsistent bounds tightness with no stated contract (+1 more)

### Community 75 - "Community 75"
Cohesion: 0.08
Nodes (16): square_in_rect(), _brute_force_sdf(), method='centroid' on a shape that Shapely can represent should work., TestInfiniteBoundsError, Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha (+8 more)

### Community 77 - "Community 77"
Cohesion: 0.06
Nodes (40): _centroid(), TestBooleansToShapely, TestPrimitivesToShapely, Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, Shapely Adapter Pattern, difference_to_shapely() (+32 more)

### Community 80 - "Community 80"
Cohesion: 0.67
Nodes (3): Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, SHAPE_SAMPLER_REGISTRY

### Community 84 - "Community 84"
Cohesion: 0.40
Nodes (5): Post-screening findings (from the dtype/device/gradient test pass), S-22 — `IsoscelesTrapezoid` never got the S-02 treatment ✅ Fixed, S-23 — `Ellipse`/`Egg`'s cubic-solve has a real cube-root gradient singularity ✅ Fixed, S-24 — Branch-1 `acos(±1)` gradient singularity on any on-axis query ✅ Fixed, S-25 — Near-circular ellipses overflow `c³` in float32, producing NaN *values* ✅ Fixed

### Community 85 - "Community 85"
Cohesion: 0.40
Nodes (4): Screening: `src/metashapes/shape/` + `src/metashapes/lattice/`, Severity / Priority / Effort, Suggested fix order (P0 → P3), Summary

## Knowledge Gaps
- **61 isolated node(s):** `allow`, `PreToolUse`, `code:bash (graphify query "<your question>"       # any codebase questi)`, `Overview`, `code:bash (source .venv/bin/activate)` (+56 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Shape` connect `Random Generator & Lattice` to `Shape Primitives Core`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `Conic Shape Primitives`, `Shapely Transform Tests`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Generator Base Classes`, `Community 11`, `Shapely Adapter Modules`, `Unit Cell Tests`, `Shapely Adapter Layer`, `Lattice SDF Tests`, `Coordinate Transform Bridge`, `Periodic Unit Cell Ops`, `Shape Test Init`, `Lattice Test Init`, `Adapters Test Init`, `Lattice Basis Rationale`, `Lattice Basis Rationale B`, `Validator Rationale`, `Generator Base Rationale`, `Community 52`, `Community 55`, `Community 61`, `Community 69`, `Community 71`, `Community 77`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `Rectangle` connect `Community 61` to `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Shapely Transform Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Shapely Adapter Modules`, `YAML & Unit Cell Serialization`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Boolean Shape Tests`, `Community 24`, `Community 25`, `Shape Test Init`, `Lattice Test Init`, `Adapters Test Init`, `Lattice Basis Rationale`, `Community 53`, `Community 55`, `Community 65`, `Community 69`, `Community 75`, `Community 77`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `Lattice` connect `Unit Cell Analyzer` to `Shape Primitives Core`, `Shape Analysis & SDF Concepts`, `YAML Serialization Tests`, `Shapely Transform Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Community 75`, `Community 11`, `YAML & Unit Cell Serialization`, `PyTorch Differentiability`, `Boolean Shape Tests`, `Community 53`, `Community 55`, `Community 24`, `Lattice Basis Rationale`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Are the 84 inferred relationships involving `UnitCell` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`UnitCell` has 84 INFERRED edges - model-reasoned connections that need verification._
- **Are the 102 inferred relationships involving `Rectangle` (e.g. with `Shape` and `TestLeafShapes`) actually correct?**
  _`Rectangle` has 102 INFERRED edges - model-reasoned connections that need verification._
- **Are the 89 inferred relationships involving `Lattice` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`Lattice` has 89 INFERRED edges - model-reasoned connections that need verification._
- **Are the 67 inferred relationships involving `Ellipse` (e.g. with `Shape` and `CrossSampler`) actually correct?**
  _`Ellipse` has 67 INFERRED edges - model-reasoned connections that need verification._