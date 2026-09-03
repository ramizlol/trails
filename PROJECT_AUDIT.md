# Trail Running Creator — Project Audit

Audit branch: `codex-project-audit`  
Baseline: `main` at `017da8f107e34bec0b764e983abd8475e27d0423`  
Audit date: 2026-09-02

## Product constraints to preserve

- Route only over real OpenStreetMap trail/walkable segments; never invent trail geometry.
- Keep the current start/end input workflow and interactive map.
- Treat distance, elevation gain, vertical density, and terrain as first-class route targets.
- Preserve trail-focused routing, pass points, avoid/prefer controls, section replacement, route alternatives, and COROS-compatible GPX without elevation.
- Keep normal production routing independent of a large runtime DEM.
- Expand coverage across Arizona, with an architecture that can later extend beyond Arizona.
- Protect the currently working `main` branch until changes are tested.

## Current state

### What is already strong

- The application uses local OSM data and does not depend on Overpass for normal production requests.
- Runtime routing is geographically tiled, limiting how much NetworkX data is loaded per request.
- Lightweight overlay tiles avoid loading routing graphs merely to draw the map.
- Trail elevation profiles are derived from the 10 m USGS DEM at build time and embedded compactly in routing tiles.
- Rustworkx accelerates repeated shortest-path operations.
- The route engine already supports loops, point-to-point routes, pass points, avoid areas, exact trail-segment controls, section replacement, alternatives, diversity, GPX export, and detailed quality metrics.
- Current `main.py` passes Python syntax compilation.
- Previously observed defects involving `H` in compact elevation handling, `selected_u/selected_v`, and `PathMapping.get` are fixed in the current snapshot.

### Repository facts

- `main.py`: 13,543 lines / approximately 482 KB.
- Inline `route_map()` frontend: 4,179 lines / approximately 147 KB.
- 195 Python functions and 8 classes are defined in `main.py`.
- 51 broad `except Exception` handlers appear in `main.py`.
- Routing data: 69 tiles, 66 non-empty, approximately 102 MB compressed on disk.
- Tile manifest totals: 48,527 nodes, 107,658 directed edges, 98,844 trail edges, and 8,814 connector edges before overlap deduplication.
- Overlay data: 49,219 trail segments and approximately 15 MB.
- Nominal coverage is `32.75–34.75 N`, `113.25–110.25 W`; this is central Arizona, not statewide.
- The repository has no automated tests, CI workflow, README, deployment manifest, lockfile, or explicit Python version.

## Prioritized improvements

### P0 — Correctness and production reliability

1. **Build a regression and benchmark suite before major refactoring**
   - Add unit tests using small synthetic MultiDiGraphs.
   - Add integration tests for exact start/end insertion, loop closure, tile boundaries, parallel edges, connectors, pass points, avoid/prefer controls, section replacement, GPX export, and deterministic seeds.
   - Add golden GPX benchmarks for short, medium, long, and ultra routes.
   - Track distance error, gain error, retrace, connector use, trail percentage, route diversity, runtime, and peak memory.
   - Run the suite in GitHub Actions on every branch and pull request.

2. **Repair GPX analysis in DEM-free production**
   - `/analyze-gpx` and `/test-gpx` currently call `elevations_for_coords()`, which requires `output_hh.tif`; that TIFF was intentionally removed from runtime.
   - Map-match GPX points to routing edges and interpolate from the embedded 10 m-derived elevation profiles.
   - Extend `/test-gpx` beyond its current under-4-mile limitation.

3. **Preserve exact edge identity through the routing pipeline**
   - Rustworkx returns node paths, after which geometry, gain, repetition, and preference metrics reselect an edge with `get_shortest_edge()`.
   - Parallel OSM edges can therefore be routed on one edge but measured or displayed using another.
   - Carry `(u, v, key)` edge sequences, or store the chosen edge key in the simple/rustworkx graph payload and reconstruct the exact routed edge sequence.
   - Use those identities consistently for geometry, elevation, connector totals, repetition, preferences, signatures, edits, and GPX export.

4. **Insert and validate the requested end point**
   - The start can be split into the exact selected trail edge, but point-to-point routing currently snaps the end to the nearest node without comparable validation.
   - Insert the end on the exact trail edge, report end snap distance, reject unreachable/out-of-coverage endpoints, and display the requested-versus-routed end in the UI.
   - Replace the implicit approximately-11-meter “same point means loop” rule with an explicit route type.

5. **Make shared caches and seeded searches concurrency-safe**
   - `random.seed()` changes process-global state, so simultaneous requests can interfere with reproducibility.
   - Evicting a workspace calls `old_workspace.clear()`, which can mutate an object still in use by another request.
   - Use request-local `random.Random(seed)`, immutable/read-only cached workspaces, and safe LRU eviction without clearing live objects.
   - Add concurrency tests for different starts, distances, and seeds.

6. **Add strict input and resource validation**
   - Validate latitude/longitude, finite numeric values, target limits, pass/avoid tolerances, route-edit array sizes, and coordinate counts with Pydantic constraints.
   - Enforce GPX upload byte and point limits before densifying at 5 m.
   - Bound expensive route-search requests and return actionable 4xx errors.
   - Do not expose raw exception strings as public 500 responses.

7. **Validate routing artifacts at startup and load time**
   - Verify manifest schema, elevation-storage version, tile schema, per-tile DEM signature, required attributes, size, and checksum.
   - Fail clearly on partial, stale, corrupt, or library-incompatible pickle data.
   - Make `point_inside_dem()` fail closed when coverage metadata cannot be read.
   - Add `/health/live` and `/health/ready` checks; readiness should verify manifests and sample tiles without loading the entire state.

8. **Fix incremental tile-build manifest behavior**
   - Running `build_routing_tiles.py --tile ...` currently writes a manifest containing only the selected tile.
   - Skipping existing tiles reconstructs rows without all original metadata fields.
   - Merge updated rows into the existing manifest atomically, preserve untouched rows, and validate completeness before replacing the manifest.
   - Build overlay manifests only from a successfully validated routing manifest.

### P1 — Route quality

9. **Turn GPX examples into an objective route-quality harness**
   - Test real trails across multiple network shapes: dense loops, ridge out-and-backs, sparse desert systems, connector-dependent systems, and high-vertical mountain terrain.
   - Compare whether the generator finds a viable route near the same target—not whether it reproduces a specific runner’s exact track.
   - Establish quality gates by distance tier rather than tuning constants by anecdote.

10. **Calibrate elevation gain**
    - The source is a 10 m DEM, sampled every 5 m, smoothed over approximately 55 m, then every positive change is summed.
    - Validate gain against known GPX/barometric references and tune smoothing or a vertical-noise/hysteresis rule.
    - Version the elevation algorithm separately from the tile format so quality changes are measurable and reproducible.

11. **Make point-to-point routing honor distance and gain**
    - The current point-to-point path is essentially the shortest routing-cost path through required points; target distance and target gain do not shape the path.
    - Add constrained detour/waypoint search for point-to-point requests.
    - Either preserve user-specified pass-point order or optimize order using routed cost, not greedy straight-line distance.

12. **Improve route ranking and visible diversity**
    - The API and browser sort alternatives primarily by ascending mileage, which can put the best target match behind a worse option.
    - Default to the best quality score/target match, with optional sorting by distance, gain, retrace, trail percentage, or reach.
    - Keep diversity soft, but add a minimum meaningful-difference indicator and cluster nearly identical alternatives.

13. **Use trail difficulty and terrain attributes**
    - `sac_scale`, `trail_visibility`, `tracktype`, and `surface` are loaded but mostly unused for route suitability.
    - Add difficulty, technicality, surface, grade, vertical-per-mile, and connector-tolerance controls.
    - Exclude private/no-foot access and clearly flag permissive, poorly visible, highly technical, or potentially unsuitable segments.
    - Preserve OSM provenance so the user can inspect why a segment was included.

14. **Audit connectors and tile-boundary connectivity**
    - Connector counts are capped per tile with hard-coded limits; this can leave useful systems disconnected or favor arbitrary component pairs.
    - Validate graph connectivity across every adjacent tile border.
    - Score connectors using real OSM walkability, access, length, road class, and crossing safety.
    - Ensure every displayed route segment traces an OSM geometry; never substitute a straight-line gap.

15. **Make route editing use the same quality and identity model**
    - Section replacement, recalculate, avoid/prefer, and generated routes should all use exact edge identities and the same elevation/repetition logic.
    - Add undo/redo and edit regression tests so edits cannot silently introduce off-network geometry or inconsistent metrics.

### P1 — Statewide Arizona and future scaling

16. **Replace hard-coded central-Arizona tiling with a reusable pipeline**
    - Generate a grid from a configured state/region polygon.
    - Automatically subdivide dense tiles based on node/edge count or memory estimates instead of the hard-coded `DENSE_PARENT_REPLACEMENTS` table.
    - Support resumable, incremental, parallel offline builds with deterministic outputs.
    - Record OSM extract timestamp, source URL/checksum, DEM version, code commit, build parameters, and tile checksum.

17. **Represent actual coverage, not only a nominal rectangle**
    - The nominal rectangle includes empty or partially covered cells and does not describe the real DEM/OSM footprint.
    - Publish an actual coverage polygon plus tile availability/health.
    - Let the map show covered, build-pending, and unavailable regions.
    - Do not count overlapping parent/child tile contents as unique statewide totals.

18. **Create offline trail discovery and regional browsing**
    - Build an index of trail names, trailheads, parks, regions, OSM identifiers, approximate length, elevation range, technical attributes, and connectivity.
    - Add name/place search, “trails near here,” and regional browsing without changing the existing coordinate-based start/end workflow.
    - Use the same index to suggest good starting points for a requested distance/gain.

19. **Separate derived geographic data from application source**
    - The repository is already approximately 267 MB and contains current tiles, overlays, legacy master pickles, a Phoenix PBF, and an old 30 m TIFF.
    - Remove obsolete duplicate artifacts after validation.
    - Publish immutable versioned data bundles with checksums using release/object storage or a dedicated data repository.
    - Let production download/cache only the manifest and required regional tiles, while retaining predictable cold-start behavior.

### P2 — Maintainability, deployment, security, and UX

20. **Split the 13,543-line monolith**
    - Suggested packages: `api`, `routing`, `elevation`, `graph_io`, `tile_build`, `gpx`, `models`, and `settings`.
    - Move the 4,179-line inline frontend into versioned HTML/CSS/JavaScript files.
    - Remove legacy monolithic graph/runtime code after equivalent tiled tests pass.
    - Replace V14/V15/V37/V52 comments with release notes and focused explanations of current behavior.

21. **Make builds and deployments reproducible**
    - Pin all direct and transitive dependencies with a lockfile.
    - Declare and test an explicit Python version.
    - Add a Dockerfile and Render configuration, including start command, health check, memory assumptions, and data version.
    - Add a local development command and a separate offline-data-build command.

22. **Add documentation**
    - Create a README with architecture, setup, deployment, data provenance, supported route modes, current coverage, and known limitations.
    - Document how to rebuild one tile, a region, overlays, and the full manifest safely.
    - Add an operator runbook for updating OSM/DEM data and rolling back a bad dataset.

23. **Add structured observability**
    - Replace `print()` timing reports with structured logs.
    - Record request ID, route tier, selected tiles, graph size, cache hit, search runtime, candidate count, failure reason, and memory high-water mark.
    - Add aggregate metrics for route success and quality without logging precise user routes by default.

24. **Harden the public API and frontend**
    - Use response models and stable API versioning.
    - Apply upload/request limits and rate limits to CPU-heavy endpoints.
    - Parse untrusted GPX/XML defensively.
    - Avoid inserting raw error text with `innerHTML`.
    - Self-host or integrity-pin frontend dependencies and apply a Content Security Policy.

25. **Improve route-planning UX**
    - Show a clear route-quality summary and why each alternative differs.
    - Add route sorting, comparison, coverage status, search progress, cancellation, and timeout feedback.
    - Preserve planner state in a shareable URL or saved plan.
    - Test mobile controls, keyboard access, screen-reader labels, and high-contrast route layers.

26. **Automate OSM freshness**
    - Rebuild affected tiles from timestamped OSM extracts on a controlled schedule.
    - Detect access/tag changes and validate data before publishing a new manifest.
    - Keep prior data versions available for rollback.

## Recommended implementation sequence

1. Add project structure, README, pinned environment, tests, CI, and benchmark fixtures without changing routing behavior.
2. Fix DEM-free GPX analysis, exact end handling, edge identity, concurrency, input limits, and manifest corruption risks.
3. Establish repeatable route-quality benchmarks and tune elevation, route ranking, diversity, point-to-point behavior, and connector selection.
4. Refactor the backend/frontend monolith behind passing regression tests.
5. Build the generic statewide Arizona data pipeline and actual-coverage index.
6. Add regional trail discovery, terrain controls, production observability, and automated data refresh.
7. Publish a pull request only after the branch meets the release gates below.

## Proposed release gates

- Existing start/end, map, editing, route alternatives, and COROS GPX workflows remain functional.
- No route geometry leaves real OSM trail/walkable edges except an explicitly represented user-to-trail access offset.
- Exact routed edge identity is used consistently for display and metrics.
- GPX analysis works without a runtime TIFF.
- Same seed + same dataset + same request produces the same result under concurrent load.
- No tile/manifest update can silently delete unrelated coverage.
- Short, medium, long, and ultra benchmark suites meet agreed distance/gain/retrace/runtime thresholds.
- A cold Render instance stays within its memory limit for representative routes.
- Coverage status is accurate and statewide builds are resumable and reproducible.
- `main` remains untouched until the audit branch passes CI and is reviewed.
