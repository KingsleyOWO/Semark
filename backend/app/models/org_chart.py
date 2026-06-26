"""
Organizational Chart data structures.

Defines Graph-based representation for org charts:
- nodes: organizational units
- edges: relationships between units
- groups: functional categories
- derived_paths: PATH notation for RAG retrieval
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OrgCategory(StrEnum):
    """Functional category for organizational units."""
    GOVERNANCE = "治理監督"      # 董事會、監察人、稽核室
    MANAGEMENT = "經營管理"      # 院長、副院長、主任秘書
    COMMITTEE = "委員會"         # 各種委員會
    DEPARTMENT = "正式編制"      # 研究所、處、室
    TASKFORCE = "任務編組"       # 研究中心、辦公室
    UNKNOWN = "未分類"


class EdgeType(StrEnum):
    """Relationship type between organizational units."""
    REPORTS_TO = "reports_to"       # 直接匯報（實線）
    OVERSIGHT = "oversight"          # 監督關係
    ADVISORY = "advisory"            # 諮詢關係（虛線）
    SIBLING = "sibling"              # 平行單位（同層）
    CONTAINS = "contains"            # 包含（如院長室下轄中心）
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    """Provenance source for nodes/edges."""
    MINERU_TEXT_BLOCK = "mineru_text_block"
    HEURISTIC = "heuristic"
    VLM = "vlm"
    CV = "cv"  # Future: OpenCV detection


@dataclass
class Evidence:
    """Provenance evidence for traceability."""
    page_idx: int
    bbox_norm: list[float] | None = None
    asset_path: str | None = None
    block_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_idx": self.page_idx,
            "bbox_norm": self.bbox_norm,
            "asset_path": self.asset_path,
            "block_id": self.block_id,
        }


@dataclass
class ParentCandidate:
    """A candidate parent with score."""
    id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": self.score}


@dataclass
class EdgeCandidate:
    """
    D3: Edge candidate for VLM selection.

    Heuristic generates candidates; VLM#2 selects from them.
    """
    child_id: str
    child_label: str
    parent_candidates: list[tuple[str, str, float]]  # [(id, label, score), ...]

    def to_vlm_format(self) -> dict[str, Any]:
        """Format for VLM prompt (no IDs, only labels)."""
        return {
            "child": self.child_label,
            "candidates": [
                {"parent": label, "score": round(score, 2)}
                for _, label, score in self.parent_candidates
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Full format for debugging."""
        return {
            "child_id": self.child_id,
            "child_label": self.child_label,
            "parent_candidates": [
                {"id": pid, "label": label, "score": score}
                for pid, label, score in self.parent_candidates
            ],
        }


@dataclass
class OrgNode:
    """A node in the organizational chart."""
    id: str
    label: str
    bbox: list[float]  # [x0, y0, x1, y1] normalized
    page_idx: int

    # Classification (multi-stage)
    category_hint: OrgCategory = OrgCategory.UNKNOWN  # From text rules (weak)
    category: OrgCategory = OrgCategory.UNKNOWN       # Final (may be VLM-refined)
    hint_confidence: float = 0.5
    confidence: float = 1.0

    # Geometric info for heuristics
    center_x: float = 0.0
    center_y: float = 0.0
    level: int = -1  # Y-based level (0 = top)

    # Parent selection (for VLM constraint)
    candidate_parents: list[ParentCandidate] = field(default_factory=list)
    chosen_parent: str | None = None
    parent_decision: str = "heuristic"  # "heuristic" | "vlm" | "unknown"

    # Provenance
    source: SourceType = SourceType.MINERU_TEXT_BLOCK
    evidence: Evidence | None = None

    def __post_init__(self):
        if self.bbox and len(self.bbox) == 4:
            self.center_x = (self.bbox[0] + self.bbox[2]) / 2
            self.center_y = (self.bbox[1] + self.bbox[3]) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "bbox": self.bbox,
            "page_idx": self.page_idx,
            "category_hint": self.category_hint.value if isinstance(self.category_hint, OrgCategory) else self.category_hint,
            "category": self.category.value if isinstance(self.category, OrgCategory) else self.category,
            "hint_confidence": self.hint_confidence,
            "confidence": self.confidence,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "level": self.level,
            "candidate_parents": [c.to_dict() for c in self.candidate_parents],
            "chosen_parent": self.chosen_parent,
            "parent_decision": self.parent_decision,
            "source": self.source.value if isinstance(self.source, SourceType) else self.source,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass
class OrgEdge:
    """An edge (relationship) in the organizational chart."""
    from_id: str
    to_id: str
    edge_type: EdgeType = EdgeType.UNKNOWN
    confidence: float = 0.5
    needs_review: bool = False

    # Provenance
    source: SourceType = SourceType.HEURISTIC
    decision: str = "heuristic"  # "heuristic" | "vlm" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_id,
            "to": self.to_id,
            "type": self.edge_type.value if isinstance(self.edge_type, EdgeType) else self.edge_type,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "source": self.source.value if isinstance(self.source, SourceType) else self.source,
            "decision": self.decision,
        }


@dataclass
class OrgGroup:
    """A functional group of organizational units."""
    name: str  # Category name
    members: list[str] = field(default_factory=list)  # Node IDs
    confidence: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "members": self.members,
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class OrgChartGraph:
    """
    Complete organizational chart representation.

    This is the primary output format for org charts, replacing PATH notation.
    PATH notation is derived from this graph for backward compatibility.
    """
    nodes: list[OrgNode] = field(default_factory=list)
    edges: list[OrgEdge] = field(default_factory=list)
    groups: list[OrgGroup] = field(default_factory=list)
    derived_paths: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    # Metadata
    title: str = ""
    date: str = ""
    page_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "date": self.date,
            "page_idx": self.page_idx,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "groups": [g.to_dict() for g in self.groups],
            "derived_paths": self.derived_paths,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }

    def to_structured_content(self) -> str:
        """
        Generate RAG-friendly markdown from the graph.

        Strategy (Schema-first, validate-and-degrade):
        1. Always validate edges quality first
        2. If validation fails → degrade to groups + summary (stable output)
        3. Only render tree if edges pass validation

        Note: Title/date are NOT included here as they're added by package.py
        """
        # Always validate first
        is_valid, warnings = self.validate_edges()

        if not is_valid or not self.edges:
            # Degrade: only output groups + summary (stable, no wrong tree)
            self.needs_review = True
            self.review_reasons.extend(warnings)
            return self._generate_degraded_content(warnings)

        # Edges passed validation → render tree
        return self._generate_tree_content()

    def validate_edges(self) -> tuple[bool, list[str]]:
        """
        Validate edge quality with hard rules.

        Hard rules (hit ANY one = validation fails):
        1. root_count > 3 (too many roots indicates broken structure)
        2. max_out_degree / node_count > 0.5 (flattening: one parent has too many children)
        3. unknown_edge_ratio > 0.2 (too many unknown edges)
        4. cycle_detected == true (cycles in hierarchy)
        5. edge_count < node_count - root_count (edges too few, tree incomplete)

        Returns:
            (is_valid, warnings): tuple of validation result and warning messages
        """
        warnings: list[str] = []

        if not self.nodes:
            return False, ["No nodes found"]

        if not self.edges:
            return False, ["No edges found - using grouped output"]

        node_count = len(self.nodes)

        # Only consider hierarchical edges (REPORTS_TO, CONTAINS)
        hier_edges = [e for e in self.edges if e.edge_type in (EdgeType.REPORTS_TO, EdgeType.CONTAINS)]

        if not hier_edges:
            return False, ["No hierarchical edges found"]

        # Build parent -> children mapping
        children_map: dict[str, list[str]] = {}
        has_parent: set[str] = set()

        for edge in hier_edges:
            parent_id = edge.to_id
            child_id = edge.from_id
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
            has_parent.add(child_id)

        # Rule 1: root_count > 3
        all_node_ids = {n.id for n in self.nodes}
        root_ids = all_node_ids - has_parent
        root_count = len(root_ids)

        if root_count > 3:
            warnings.append(f"Too many roots: {root_count} (threshold: 3)")

        # Rule 2: max_out_degree / node_count > 0.5 (flattening detection)
        if children_map:
            max_out_degree = max(len(children) for children in children_map.values())
            flatten_ratio = max_out_degree / node_count if node_count > 0 else 0

            if flatten_ratio > 0.5:
                warnings.append(
                    f"Flattening detected: one parent has {max_out_degree}/{node_count} "
                    f"children ({flatten_ratio:.1%} > 50%)"
                )

        # Rule 3: unknown_edge_ratio > 0.2
        unknown_edges = [e for e in self.edges if e.edge_type == EdgeType.UNKNOWN]
        unknown_ratio = len(unknown_edges) / len(self.edges) if self.edges else 0

        if unknown_ratio > 0.2:
            warnings.append(
                f"Too many unknown edges: {len(unknown_edges)}/{len(self.edges)} "
                f"({unknown_ratio:.1%} > 20%)"
            )

        # Rule 4: cycle detection
        if self._has_cycle(hier_edges):
            warnings.append("Cycle detected in hierarchy")

        # Rule 5: edge_count < node_count - root_count (tree incomplete)
        expected_edges = node_count - root_count
        if len(hier_edges) < expected_edges * 0.5:  # Allow some tolerance
            warnings.append(
                f"Too few edges: {len(hier_edges)} < {expected_edges} expected "
                f"(nodes={node_count}, roots={root_count})"
            )

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def _has_cycle(self, edges: list["OrgEdge"]) -> bool:
        """Detect cycles in hierarchical edges using DFS."""
        # Build adjacency: child -> parent
        parent_map: dict[str, str] = {}
        for edge in edges:
            parent_map[edge.from_id] = edge.to_id

        # DFS to detect cycle
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in rec_stack:
                return True  # Cycle found
            if node_id in visited:
                return False

            visited.add(node_id)
            rec_stack.add(node_id)

            parent = parent_map.get(node_id)
            if parent and dfs(parent):
                return True

            rec_stack.remove(node_id)
            return False

        for node_id in parent_map:
            if dfs(node_id):
                return True

        return False

    def _generate_degraded_content(self, warnings: list[str]) -> str:
        """
        Generate degraded output when edges validation fails.

        Output format:
        1. Groups by category (stable, always works)
        2. Organization summary (RAG-friendly text description)
        3. Validation warnings (for debugging)
        """
        lines = []

        # === Section 1: Groups by category ===
        lines.append(self._generate_groups_content())

        # === Section 2: Organization summary ===
        lines.append("## 組織說明")
        lines.append("")

        # Generate summary based on groups
        summary_parts = []

        # Count by category
        category_counts: dict[str, int] = {}
        for group in self.groups:
            if group.members:
                category_counts[group.name] = len(group.members)

        if "治理監督" in category_counts:
            summary_parts.append(f"治理監督單位 {category_counts['治理監督']} 個")
        if "經營管理" in category_counts:
            summary_parts.append(f"經營管理層級 {category_counts['經營管理']} 個")
        if "正式編制" in category_counts:
            summary_parts.append(f"正式編制單位 {category_counts['正式編制']} 個")
        if "任務編組" in category_counts:
            summary_parts.append(f"任務編組 {category_counts['任務編組']} 個")
        if "委員會" in category_counts:
            summary_parts.append(f"委員會 {category_counts['委員會']} 個")

        if summary_parts:
            lines.append(f"本組織共有：{'、'.join(summary_parts)}。")
        else:
            lines.append(f"本組織共有 {len(self.nodes)} 個單位。")

        lines.append("")

        # === Section 3: Validation notes (for debugging, but useful for RAG too) ===
        if warnings:
            lines.append("## 備註")
            lines.append("")
            lines.append("本組織架構圖的層級關係尚待確認，僅提供分類參考。")
            lines.append("")

        return "\n".join(lines)

    def _generate_tree_content(self) -> str:
        """Generate hierarchical tree view when edges are available.

        Strategy:
        1. Administrative tree: Only REPORTS_TO/CONTAINS edges (行政隸屬)
        2. Oversight section: OVERSIGHT edges listed separately (治理監督)
        3. Advisory section: ADVISORY edges listed separately (諮詢關係)
        """
        lines = []

        # Separate edges by type
        hier_edges = [e for e in self.edges if e.edge_type in (EdgeType.REPORTS_TO, EdgeType.CONTAINS)]
        oversight_edges = [e for e in self.edges if e.edge_type == EdgeType.OVERSIGHT]
        advisory_edges = [e for e in self.edges if e.edge_type == EdgeType.ADVISORY]

        # Build parent -> children mapping (only hierarchical edges)
        children_map: dict[str, list[str]] = {}
        has_parent: set[str] = set()

        for edge in hier_edges:
            parent_id = edge.to_id
            child_id = edge.from_id
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)
            has_parent.add(child_id)

        # Find root nodes (exclude oversight/advisory sources from being roots)
        all_node_ids = {n.id for n in self.nodes}
        oversight_sources = {e.from_id for e in oversight_edges}
        advisory_sources = {e.from_id for e in advisory_edges}

        # Root = in hier tree but no parent, or not in any edge
        root_ids = all_node_ids - has_parent - oversight_sources - advisory_sources

        # Sort roots by level then by label
        root_nodes = [n for n in self.nodes if n.id in root_ids]
        root_nodes.sort(key=lambda n: (n.level, n.label))

        # === Section 1: Administrative hierarchy ===
        lines.append("## 組織架構")
        lines.append("")

        # Recursively print tree (4-space indent for CommonMark compatibility)
        def print_tree(node_id: str, indent: int = 0):
            node = self._find_node(node_id)
            if not node:
                return

            prefix = "    " * indent  # 4 spaces per level
            line = f"{prefix}- {node.label}"
            lines.append(line)

            # Get children and sort by level/label
            child_ids = children_map.get(node_id, [])
            child_nodes = [(cid, self._find_node(cid)) for cid in child_ids]
            child_nodes = [(cid, n) for cid, n in child_nodes if n]
            child_nodes.sort(key=lambda x: (x[1].level, x[1].label))

            for child_id, _ in child_nodes:
                print_tree(child_id, indent + 1)

        for root in root_nodes:
            print_tree(root.id)

        lines.append("")

        # Collect nodes in tree for orphan detection
        tree_nodes: set[str] = set()
        def collect_tree_nodes(node_id: str):
            tree_nodes.add(node_id)
            for child_id in children_map.get(node_id, []):
                collect_tree_nodes(child_id)
        for root in root_nodes:
            collect_tree_nodes(root.id)

        # === Section 2: Oversight relationships (non-hierarchical) ===
        if oversight_edges:
            lines.append("## 治理監督")
            lines.append("")
            for edge in oversight_edges:
                from_node = self._find_node(edge.from_id)
                to_node = self._find_node(edge.to_id)
                if from_node and to_node:
                    lines.append(f"- {from_node.label}（監督：{to_node.label}）")
                    tree_nodes.add(edge.from_id)  # Mark as accounted for
            lines.append("")

        # === Section 3: Advisory relationships (committees) ===
        if advisory_edges:
            lines.append("## 委員會")
            lines.append("")
            for edge in advisory_edges:
                from_node = self._find_node(edge.from_id)
                to_node = self._find_node(edge.to_id)
                if from_node and to_node:
                    lines.append(f"- {from_node.label}（諮詢：{to_node.label}）")
                    tree_nodes.add(edge.from_id)  # Mark as accounted for
            lines.append("")

        # === Section 4: Orphan nodes (not in any relationship) ===
        orphan_nodes = [n for n in self.nodes if n.id not in tree_nodes]
        if orphan_nodes:
            lines.append("## 其他單位")
            lines.append("")
            for node in orphan_nodes:
                lines.append(f"- {node.label}")
            lines.append("")

        return "\n".join(lines)

    def _generate_groups_content(self) -> str:
        """Generate groups-based listing when no edges available."""
        lines = []

        # Category descriptions for RAG context
        category_descriptions = {
            "治理監督": "最高權力與監督機構",
            "經營管理": "經營管理層級",
            "委員會": "跨部門委員會，負責特定院務",
            "正式編制": "正式編制單位，平行一級單位",
            "任務編組": "任務編組，專案性質",
            "未分類": "待分類單位",
        }

        # Groups with members (sorted by hierarchy level)
        group_order = ["治理監督", "經營管理", "委員會", "正式編制", "任務編組", "未分類"]
        sorted_groups = sorted(
            self.groups,
            key=lambda g: group_order.index(g.name) if g.name in group_order else 99
        )

        for group in sorted_groups:
            if not group.members:
                continue
            # Skip 未分類 if empty or minimal
            if group.name == "未分類" and len(group.members) <= 1:
                continue

            lines.append(f"## {group.name}")
            # Add description
            desc = group.description or category_descriptions.get(group.name, "")
            if desc:
                lines.append(desc)

            # Find nodes in this group
            for node_id in group.members:
                node = self._find_node(node_id)
                if node:
                    lines.append(f"- **{node.label}**")
            lines.append("")

        return "\n".join(lines)

    def _find_node(self, node_id: str) -> OrgNode | None:
        """Find node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_roots(self) -> list[OrgNode]:
        """Get root nodes (nodes with no incoming reports_to edges)."""
        child_ids = {e.from_id for e in self.edges if e.edge_type == EdgeType.REPORTS_TO}
        return [n for n in self.nodes if n.id not in child_ids]

    def get_children(self, node_id: str) -> list[OrgNode]:
        """Get direct children of a node."""
        child_ids = {e.from_id for e in self.edges
                     if e.to_id == node_id and e.edge_type == EdgeType.REPORTS_TO}
        return [n for n in self.nodes if n.id in child_ids]
