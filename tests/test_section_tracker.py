from app.models import SectionTracker

def test_section_path_carries_across_pages():
    tracker = SectionTracker()

    # Page 3: section starts
    path = tracker.update("## 2. Methodology\nSome text.", 3)
    assert path == ["2. Methodology"]

    # Pages 4-6: no new headings — path must persist
    for page in [4, 5, 6]:
        path = tracker.update("Continued text with no headings.", page)
        assert path == ["2. Methodology"], f"Path lost on page {page}"

    # Page 7: subsection
    path = tracker.update("### 2.1 Setup\nText.", 7)
    assert path == ["2. Methodology", "2.1 Setup"]

    # Page 8: new top-level section — stack should pop 2. Methodology
    path = tracker.update("## 3. Results\nText.", 8)
    assert path == ["3. Results"]

def test_section_tree_built_correctly():
    tracker = SectionTracker()
    tracker.update("# Introduction\n## 1.1 Background", 1)
    tracker.update("## 1.2 Related Work", 2)
    tree = tracker.get_tree()
    assert len(tree) == 3
    assert tree[0].level == 1
    assert tree[1].level == 2