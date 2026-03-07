from app.pipeline.vector_detect import _cluster_drawings

def make_drawing(x0, y0, x1, y1):
    return {"rect": [x0, y0, x1, y1]}

def test_adjacent_rects_merge():
    drawings = [
        make_drawing(0, 0, 50, 50),
        make_drawing(55, 0, 100, 50),   # 5px gap — within tolerance
    ]
    clusters = _cluster_drawings(drawings, gap_tolerance=20)
    assert len(clusters) == 1

def test_distant_rects_stay_separate():
    drawings = [
        make_drawing(0,   0, 50,  50),
        make_drawing(200, 0, 250, 50),  # far apart
    ]
    clusters = _cluster_drawings(drawings, gap_tolerance=20)
    assert len(clusters) == 2

def test_non_sequential_order_merges_correctly():
    # Simulates drawings that arrive out of left-to-right order
    # (common in complex PDFs) — the naive sweep fails this
    drawings = [
        make_drawing(0,   0, 50,  50),
        make_drawing(200, 0, 250, 50),  # far — separate cluster
        make_drawing(55,  0, 100, 50),  # close to first — should merge with first
    ]
    clusters = _cluster_drawings(drawings, gap_tolerance=20)
    assert len(clusters) == 2
    xs = sorted(c[0] for c in clusters)
    assert xs[0] == 0      # merged first+third
    assert xs[1] == 200    # standalone