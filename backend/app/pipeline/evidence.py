"""Immutable, source-bound observations. Extractor confidence never certifies content."""
import hashlib
import json
import shutil
from pathlib import Path

from app.pipeline.delivery import atomic_write_json
from app.pipeline.review_sources import digest


class EvidenceStore:
    def __init__(self, root, source_sha):
        self.root = Path(root)
        self.source_sha = source_sha
        self.directory = self.root / "agentic/evidence/observations"

    def add(self, *, provider, kind, page, block_id=None, source=None, bbox=None, content=None,
            metadata=None, parents=()):
        self.directory.mkdir(parents=True, exist_ok=True)
        source_ref = None
        if source is not None:
            source = Path(source)
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            target = self.directory / "sources" / (sha + source.suffix.lower())
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            elif hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise ValueError("evidence_source_changed")
            source_ref = {"path": str(target.relative_to(self.root)), "sha256": sha}
        for parent in parents:
            self.get(parent, page=page, block_id=block_id)
        record = {"version": 1, "document_sha256": self.source_sha, "provider": provider,
                  "kind": kind, "page_idx": page, "block_id": block_id, "source": source_ref,
                  "bbox": bbox, "content": content, "metadata": metadata or {},
                  "parents": list(parents), "status": "observation"}
        eid = "ev_" + digest(record)
        path = self.directory / (eid + ".json")
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("evidence_record_changed")
        atomic_write_json(path, record, ensure_ascii=False, indent=2)
        return eid

    def get(self, eid, *, page=None, block_id=None):
        if not isinstance(eid, str) or not eid.startswith("ev_") or not all(
                c in "0123456789abcdef" for c in eid[3:]) or len(eid) != 67:
            raise ValueError("unknown_evidence_id")
        path = self.directory / (eid + ".json")
        if not path.is_file():
            raise ValueError("unknown_evidence_id")
        value = json.loads(path.read_text())
        if "ev_" + digest(value) != eid or value["document_sha256"] != self.source_sha:
            raise ValueError("evidence_identity_changed")
        if page is not None and value["page_idx"] != page:
            raise ValueError("evidence_page_mismatch")
        if block_id is not None and value["block_id"] not in {None, block_id}:
            raise ValueError("evidence_block_mismatch")
        if source := value.get("source"):
            original = self.root / source["path"]
            if (not original.resolve().is_relative_to(self.directory.resolve()) or not original.is_file()
                    or hashlib.sha256(original.read_bytes()).hexdigest() != source["sha256"]):
                raise ValueError("evidence_source_changed")
        for parent in value.get("parents", []):
            self.get(parent, page=value["page_idx"], block_id=value["block_id"])
        return value

    def for_target(self, page, block_id, *, provider=None):
        result = []
        for path in sorted(self.directory.glob("ev_*.json")):
            value = self.get(path.stem)
            if (value["page_idx"] == page and value["block_id"] == block_id
                    and (provider is None or value["provider"] == provider)):
                result.append({"evidence_id": path.stem, **value})
        return result

    def validate_candidate(self, candidate):
        for block in candidate.blocks:
            for eid in block.payload.get("agentic_evidence_ids", []):
                value = self.get(eid, block_id=block.payload.get("agentic_image_origin"))
                if value["page_idx"] not in block.payload.get("agentic_source_pages", []):
                    raise ValueError("candidate_evidence_page_mismatch")

    def copy_to(self, folder):
        if self.directory.exists():
            shutil.copytree(self.directory, Path(folder) / "agentic/evidence/observations", dirs_exist_ok=True)
