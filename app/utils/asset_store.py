"""
asset_store.py — Persistent asset storage for parsed document outputs.
One AssetStore per request. Saves PNG crops to disk permanently so the UI
can serve them after the temp working dir is deleted.
"""
import base64
from pathlib import Path
from app.config import ASSETS_BASE_DIR


class AssetStore:
    """
    Manages the persistent output directory for one document.
    Structure: {ASSETS_BASE_DIR}/{doc_id}/assets/

    Usage:
        store = AssetStore(doc_id)
        file_path = store.save_image(image_b64, "p3_figure_0.png")
        # file_path = "assets/p3_figure_0.png"  (relative)
    """

    def __init__(self, doc_id: str):
        self.doc_id   = doc_id
        self.root     = Path(ASSETS_BASE_DIR) / doc_id
        self.asset_dir = self.root / "assets"
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image_b64: str, filename: str) -> str:
        """
        Decode base64 PNG and write to asset dir.
        Returns relative path string: "assets/{filename}"
        """
        data = base64.b64decode(image_b64)
        out  = self.asset_dir / filename
        out.write_bytes(data)
        return f"assets/{filename}"

    def save_bytes(self, data: bytes, filename: str) -> str:
        """Save raw bytes directly. Returns relative path."""
        out = self.asset_dir / filename
        out.write_bytes(data)
        return f"assets/{filename}"

    def absolute_path(self, relative_path: str) -> Path:
        """Resolve a stored relative path back to absolute for serving."""
        return self.root / relative_path

    def doc_root(self) -> Path:
        return self.root