# Graph Report - metashapes  (2026-08-01)

## Corpus Check
- 74 files · ~54,452 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1384 nodes · 4571 edges · 66 communities (45 shown, 21 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 1056 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2147b755`
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

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 153 edges
2. `Rectangle` - 147 edges
3. `Lattice` - 129 edges
4. `shape_to_shapely()` - 110 edges
5. `Ellipse` - 108 edges
6. `RegularPolygon` - 99 edges
7. `Bar` - 94 edges
8. `Shape` - 87 edges
9. `assert_inside()` - 76 edges
10. `Translate` - 70 edges

## Surprising Connections (you probably didn't know these)
- `ConvexQuad` --semantically_similar_to--> `IsoscelesTrapezoid`  [INFERRED] [semantically similar]
  src/metashapes/shape/primitives/quads.py → metashapes/shape/primitives/quads.py
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → src/metashapes/shape/primitives/conics.py
- `UnitCellAnalyzer.validate` --semantically_similar_to--> `UnitCellValidator`  [INFERRED] [semantically similar]
  metashapes/analysis.py → src/metashapes/generators/validator.py
- `sdf_at()` --references--> `SDF Convention: Negative Inside, Positive Outside`  [INFERRED]
  tests/shape/conftest.py → metashapes/shape/base.py
- `TestLeafShapes` --uses--> `CellMetrics`  [INFERRED]
  tests/test_analysis.py → src/metashapes/analysis.py

## Communities (66 total, 21 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.14
Nodes (18): TestBarSafeExtraction, TestCrossSafeExtraction, TestEllipseSafeExtraction, TestIsoscelesTrapezoidSafeExtraction, TestRectangleSafeExtraction, TestRegularPolygonSafeExtraction, TestStadiumSafeExtraction, TestStripeSafeExtraction (+10 more)

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.05
Nodes (40): _centroid(), p(), pv(), Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestTransformSafeExtraction, TestBooleansToShapely, TestPrimitivesToShapely (+32 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.11
Nodes (17): _leaf_shapes, UnitCellAnalyzer, _leaf_shapes(), Compute all metrics for a single cell., Return a list of constraint violation descriptions.         An empty list means, Generator-compatible interface.          Returns the first constraint violation, Compute metrics for every cell in a batch., Find groups of cells that have identical SDFs (within tolerance).          Each (+9 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.07
Nodes (43): register_shape_sampler(), Lattice, Cartesian translation for lattice cell (i, j)., In-plane periodicity of the unit cell. Fixed (non-optimizable).     Defined by t, Cartesian (x, y) -> fractional (f1, f2)., Fractional (f1, f2) -> Cartesian (x, y)., TestLatticeConstruction, TestLatticeCoordinates (+35 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.11
Nodes (11): Ellipse, Ellipse.min_feature_size, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Inverse Design for Metasurfaces, SDF-based differentiable shapes concept, Shape + Lattice = UnitCell model, MetaShapes Project Documentation (+3 more)

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.11
Nodes (6): Rectangular unit cell with a small square at the origin., _square_cell(), TestExtent, TestUnitCellBoundary, TestUnitCellMask, TestUnitCellRasterize

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.16
Nodes (6): assert_inside(), assert_outside(), All points should have SDF < tol (inside or on boundary)., All points should have SDF > -tol (outside or on boundary)., TestCross, TestTShape

### Community 7 - "Random Generator Logic"
Cohesion: 0.10
Nodes (9): _gen(), _sdf_grid(), test_each_shape_type_generates(), TestBasicGeneration, TestConstraints, TestReport, TestShapeCount, TestShapeTypes (+1 more)

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.16
Nodes (13): GeneratorConfig, RandomUnitCellGenerator._generate_one, RandomGeneratorConfig, RandomUnitCellGenerator, First concrete generator config.      For now it only adds one flag:     - requi, First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, Random unit-cell generator.      Current responsibilities:     1. choose number (+5 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.10
Nodes (10): Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, Shape, assert_round_trip(), Serialise → deserialise and verify SDF equality on a small grid. (+2 more)

### Community 10 - "Generator Base Classes"
Cohesion: 0.12
Nodes (4): General triangle defined by two base angles and the base length (ASA).      Para, Triangle, TestTriangle, TestTriangleDtypeDeviceGrad

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (16): ABC, UnitCellAnalyzer.validate, _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator (+8 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.11
Nodes (4): Bar, An infinite bar spanning the full unit cell along one axis.      The bar is unbo, TestBar, TestBarDtypeDeviceGrad

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.29
Nodes (7): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, load_unit_cells(), Save one or more unit cells to a YAML file.      Parameters     ----------     p, Load unit cells from a YAML file saved by :func:`save_unit_cells`     or :func:`, save_unit_cells()

### Community 14 - "Unit Cell Tests"
Cohesion: 0.10
Nodes (5): Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, RegularPolygon, TestRegularPolygon, TestRegularPolygonDtypeDeviceGrad

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.15
Nodes (5): IsoscelesTrapezoid, Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), TestIsoscelesTrapezoid, TestIsoscelesTrapezoidDtypeDeviceGrad

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.16
Nodes (9): _brute_force_sdf(), Independent reference: fold into the cell, then search a ring far     larger tha, Independent reference: fold into the cell, then search a ring far     larger tha, sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, sdf() after the L-02/L-03 rewrite must agree with a large brute-force     refere, _sdf_at(), test_matches_brute_force(), TestPeriodicSdfMatchesBruteForce (+1 more)

### Community 17 - "Lattice SDF Tests"
Cohesion: 0.08
Nodes (6): ConvexQuad, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, TestConvexQuad, TestConvexQuadDtypeDeviceGrad, TestRectangle

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.12
Nodes (5): Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Star, TestStar, TestStarDtypeDeviceGrad

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.19
Nodes (12): _make_batch_result(), TestSaveBatchResult, _check_version(), _Dumper, load_batch_result(), _load_cell(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Recursively convert tuples → lists so yaml.dump produces clean YAML. (+4 more)

### Community 20 - "Periodic Unit Cell Ops"
Cohesion: 0.06
Nodes (32): cell_area(), device(), dtype(), hexagonal(), matrix(), rectangular(), cartesian_grid(), fractional_grid() (+24 more)

### Community 21 - "Mask Tests"
Cohesion: 0.12
Nodes (16): Analysis, Boolean Operations, code:bash (pip install git+https://github.com/RodionovSA/Metashapes), code:python (from metashapes import UnitCell, Lattice), code:python (from metashapes.shape import Ellipse, Rectangle), code:python (from metashapes import Lattice), code:python (from metashapes import UnitCellAnalyzer), code:python (from metashapes.adapters.yaml import save_unit_cells, load_u) (+8 more)

### Community 22 - "Generator Validation"
Cohesion: 0.16
Nodes (15): gdstk_to_shapely(), klayout_to_shapely(), numpy_to_shapely(), Convert a shapely geometry to a gdstk polygon or a list of polygons., Convert a gdstk polygon to a shapely geometry., Convert a shapely geometry to a KLayout polygon or a list of polygons., Create a shapely geometry from a binary numpy array.     Parameters:         img, shapely_to_gdstk() (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.04
Nodes (47): code:block1 (Egg(center=[0,0], width=2.0, height=2.5, skew=0.6)   # a=1, ), code:python (a = torch.tensor([1.0], requires_grad=True)), code:block11 (fractional_grid(4,4) row:              [0.0, 0.25, 0.5, 0.75), code:python (from metashapes.lattice import Lattice   # ImportError), code:python (lat = Lattice.rectangular(1.0, 1.0)), code:block3 (rr=0.9: constructed OK, sdf OK), code:block4 (y=0.0000: sdf=+0.00000        (center — should be strongly n), code:block5 (Rectangle(size=[0.3, 0.8]).min_feature_size          -> 0.3) (+39 more)

### Community 24 - "Community 24"
Cohesion: 0.10
Nodes (16): _cell_center(), Rectangular cell with a Rectangle shape centred at (cx, cy)., For a symmetric Rectangle, both methods produce the same offset., For a centred Ellipse, both methods produce the same offset., The geometric midpoint (a1 + a2) / 2 as plain floats., Extract the outermost Translate dx/dy buffers from the scene., _rect_cell(), TestAlreadyCentered (+8 more)

### Community 25 - "Community 25"
Cohesion: 0.06
Nodes (21): Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, Return a new UnitCell with the scene translated to the cell centre.          The, is_empty_bounds(), Signed distance evaluated on torch tensors.         x, y can be broadcastable te, True if `bounds` (as returned by `Shape.bounds()`) is an empty/inverted box., Signed distance evaluated on torch tensors.         x, y can be broadcastable te, Alias for `sdf`, so a Shape can be called directly (`shape(x, y)`)         follo (+13 more)

### Community 26 - "Test Package Root"
Cohesion: 0.15
Nodes (12): code:bash (graphify query "<your question>"       # any codebase questi), code:bash (source .venv/bin/activate), code:block3 (src/metashapes/          # src-layout; imported as `metashap), graphify, Key Modules, Metashapes — Developer Guide, Navigating This Codebase, New Shape Primitive Addition (+4 more)

### Community 27 - "Shape Test Init"
Cohesion: 0.11
Nodes (12): _ellipse_closest_point(), min_feature_size(), Nearest point on the boundary of an axis-aligned ellipse (semi-axes     a, b, ce, Cross and TShape used to each carry their own inline copy of the     rounded-box, Cross and TShape used to each carry their own inline copy of the     rounded-box, TestSharedRoundedBoxHelper, Direct tests of the shared helper, plus a check that Rectangle's own     sdf() a, Direct tests of the shared helper, plus a check that Rectangle's own     sdf() a (+4 more)

### Community 28 - "Lattice Test Init"
Cohesion: 0.24
Nodes (11): assert_bounds_contain(), assert_direct_call_dtype_promotion(), assert_dtype_device_flow(), assert_gradients_finite(), assert_gradients_finite_at(), _assert_param_grads_finite(), Calling sdf() directly on a primitive (bypassing UnitCell) with     float64 quer, Grid-evaluate sdf() over `x_range`, sum, backward(); every named     parameter i (+3 more)

### Community 29 - "Adapters Test Init"
Cohesion: 0.13
Nodes (9): center_scene(method='centroid') must work even if params have grad., Verifies that calling shape_to_shapely (which does .detach().cpu())         does, nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., TestGradientFlow, TestOffsetSearchOverhead, Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Rectangle (+1 more)

### Community 30 - "Generators Test Init"
Cohesion: 0.14
Nodes (7): method='centroid' on a shape that Shapely can represent should work., TestInfiniteBoundsError, TestOffsetsForRing, TestToShapely, A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, UnitCell

### Community 31 - "Lattice Basis Rationale"
Cohesion: 0.15
Nodes (9): nn.Module Subclassing for Differentiability, make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., square_in_rect(), Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution. (+1 more)

### Community 32 - "Lattice Basis Rationale B"
Cohesion: 0.13
Nodes (4): Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium, TestStadium, TestStadiumDtypeDeviceGrad

### Community 35 - "Validator Rationale"
Cohesion: 0.14
Nodes (4): Egg, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, TestEgg, TestEggDtypeDeviceGrad

### Community 36 - "Generator Base Rationale"
Cohesion: 0.14
Nodes (17): Smooth Boolean Operations via Polynomial Blending, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, Tensor -> JSON/YAML-serializable Python value.     Scalar tensor -> Python scala, to_plain_data(), Difference, from_parametric(), Intersection (+9 more)

### Community 52 - "Community 52"
Cohesion: 0.10
Nodes (11): min_feature_size(), min_feature_size(), _max_corner_radius(), min_feature_size(), _quad_vertices(), _signed_area2(), register_shape(), from_parametric() (+3 more)

### Community 53 - "Community 53"
Cohesion: 0.13
Nodes (8): Parametric Serialization (to_parametric / from_parametric), SDF Convention: Negative Inside, Positive Outside, from_parametric(), min_feature_size(), Base class for all symbolic 2D shapes., Base class for all symbolic 2D shapes., Shape, SHAPE_REGISTRY Dict

### Community 54 - "Community 54"
Cohesion: 0.14
Nodes (16): CellMetrics, _compute_min_gap, Rotation Guard for Infinite-Extent Shapes, _bbox_size(), CellMetrics, _compute_min_gap(), True if this periodic shift should be skipped for self-gap measurement.      A s, Minimum distance between any two shapes (or a shape and its own periodic     ima (+8 more)

### Community 65 - "Community 65"
Cohesion: 0.67
Nodes (3): Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, SHAPE_SAMPLER_REGISTRY

## Knowledge Gaps
- **57 isolated node(s):** `allow`, `PreToolUse`, `code:bash (graphify query "<your question>"       # any codebase questi)`, `Overview`, `code:bash (source .venv/bin/activate)` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Rectangle` connect `Adapters Test Init` to `Shape Primitives Core`, `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Shapely Adapter Modules`, `YAML & Unit Cell Serialization`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Boolean Shape Tests`, `Community 24`, `Community 25`, `Shape Test Init`, `Generators Test Init`, `Lattice Basis Rationale`, `Community 52`, `Community 53`, `Community 54`, `Community 61`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `Shape` connect `Community 53` to `Shape Primitives Core`, `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `Conic Shape Primitives`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Generator Base Classes`, `Community 11`, `Shapely Adapter Modules`, `Unit Cell Tests`, `Shapely Adapter Layer`, `Lattice SDF Tests`, `Coordinate Transform Bridge`, `Periodic Unit Cell Ops`, `Community 25`, `Shape Test Init`, `Lattice Test Init`, `Adapters Test Init`, `Lattice Basis Rationale`, `Lattice Basis Rationale B`, `Validator Rationale`, `Generator Base Rationale`, `Community 52`, `Community 54`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Why does `UnitCell` connect `Generators Test Init` to `Shape Primitives Core`, `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Community 11`, `YAML & Unit Cell Serialization`, `PyTorch Differentiability`, `Boolean Shape Tests`, `Periodic Unit Cell Ops`, `Community 24`, `Community 25`, `Adapters Test Init`, `Lattice Basis Rationale`, `Community 53`, `Community 54`, `Community 60`, `Community 61`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Are the 82 inferred relationships involving `UnitCell` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`UnitCell` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 101 inferred relationships involving `Rectangle` (e.g. with `Shape` and `TestLeafShapes`) actually correct?**
  _`Rectangle` has 101 INFERRED edges - model-reasoned connections that need verification._
- **Are the 78 inferred relationships involving `Lattice` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`Lattice` has 78 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `shape_to_shapely()` (e.g. with `.test_bar_gap_is_perpendicular_only()` and `.test_bar_x_height()`) actually correct?**
  _`shape_to_shapely()` has 5 INFERRED edges - model-reasoned connections that need verification._