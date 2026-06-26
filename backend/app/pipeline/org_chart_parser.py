"""
Organizational Chart Parser (B+ Implementation).

Parses org charts using:
1. MinerU text blocks as nodes
2. Multi-stage text-based classification (hint → final)
3. Geometric heuristics for edge inference with relative thresholds
4. Candidate parent tracking for VLM constraint
5. Cycle detection and validation

Output: OrgChartGraph (nodes, edges, groups, derived_paths)
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.org_chart import (
    EdgeCandidate,
    EdgeType,
    Evidence,
    OrgCategory,
    OrgChartGraph,
    OrgEdge,
    OrgGroup,
    OrgNode,
    ParentCandidate,
    SourceType,
)

# Try to import NetworkX for cycle detection
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# Multi-stage Classification Rules
# =============================================================================

# HIGH PRIORITY rules (whitelist/keywords) - strong confidence
HIGH_PRIORITY_RULES: list[tuple[str, OrgCategory, float]] = [
    # Governance (highest priority - explicit keywords)
    (r"董事會|董事长|理事會|理事长", OrgCategory.GOVERNANCE, 0.95),
    (r"監察人|监察", OrgCategory.GOVERNANCE, 0.95),
    (r"稽核室|稽核(?!.*委員會)", OrgCategory.GOVERNANCE, 0.90),  # 稽核 but not 稽核委員會

    # Management (explicit titles)
    (r"^院長$|^院长$|執行長|执行长|總經理|总经理|CEO", OrgCategory.MANAGEMENT, 0.95),
    (r"副院長|副院长|副總經理|副总经理|COO|CFO|CTO", OrgCategory.MANAGEMENT, 0.90),
    (r"主任秘書|主任秘书|秘書長|秘书长", OrgCategory.MANAGEMENT, 0.90),
    (r"顧問室|顾问室|院長室|院长室", OrgCategory.MANAGEMENT, 0.85),  # 院長室 is management support

    # Committee (explicit keyword)
    (r"委員會|委员会", OrgCategory.COMMITTEE, 0.95),
]

# SUFFIX rules (weak) - only give hint, need VLM confirmation for low confidence
# NOTE: More specific patterns must come BEFORE general patterns!
SUFFIX_RULES: list[tuple[str, OrgCategory, float]] = [
    # Department suffixes
    (r"研究.*所|研究所", OrgCategory.DEPARTMENT, 0.85),
    (r"處$|处$", OrgCategory.DEPARTMENT, 0.75),
    (r"部$", OrgCategory.DEPARTMENT, 0.75),
    (r"資料庫$|数据库$|資訊庫$", OrgCategory.DEPARTMENT, 0.70),

    # Taskforce suffixes (more specific - must come before 室$)
    (r"辦公室$|办公室$", OrgCategory.TASKFORCE, 0.75),  # More specific than 室$
    (r"中心$", OrgCategory.TASKFORCE, 0.75),  # Research/service centers

    # Ambiguous suffixes (general - must come after specific patterns)
    (r"室$", OrgCategory.DEPARTMENT, 0.60),  # Could be 稽核室(governance) or 院長室(management)
]

# VLM refinement threshold - below this, mark for VLM review
VLM_REFINEMENT_THRESHOLD = 0.7


def classify_by_text(label: str) -> tuple[OrgCategory, OrgCategory, float]:
    """
    Multi-stage classification by text patterns.

    Returns:
        (category_hint, category, confidence)
        - category_hint: Initial guess from rules
        - category: Same as hint if confident, UNKNOWN if needs VLM
        - confidence: Rule confidence
    """
    label_clean = label.strip()

    # Try high priority rules first
    for pattern, category, confidence in HIGH_PRIORITY_RULES:
        if re.search(pattern, label_clean):
            return category, category, confidence

    # Try suffix rules (weaker)
    for pattern, category, confidence in SUFFIX_RULES:
        if re.search(pattern, label_clean):
            # If confidence < threshold, mark final as UNKNOWN for VLM review
            final_category = category if confidence >= VLM_REFINEMENT_THRESHOLD else OrgCategory.UNKNOWN
            return category, final_category, confidence

    return OrgCategory.UNKNOWN, OrgCategory.UNKNOWN, 0.3


# Edge type inference rules based on category pairs
EDGE_TYPE_RULES: dict[tuple[OrgCategory, OrgCategory], EdgeType] = {
    # Governance relations
    (OrgCategory.GOVERNANCE, OrgCategory.GOVERNANCE): EdgeType.OVERSIGHT,
    (OrgCategory.GOVERNANCE, OrgCategory.MANAGEMENT): EdgeType.OVERSIGHT,

    # Management relations
    (OrgCategory.MANAGEMENT, OrgCategory.GOVERNANCE): EdgeType.REPORTS_TO,
    (OrgCategory.MANAGEMENT, OrgCategory.MANAGEMENT): EdgeType.REPORTS_TO,
    (OrgCategory.MANAGEMENT, OrgCategory.DEPARTMENT): EdgeType.REPORTS_TO,
    (OrgCategory.MANAGEMENT, OrgCategory.TASKFORCE): EdgeType.REPORTS_TO,

    # Committee relations (advisory)
    (OrgCategory.COMMITTEE, OrgCategory.MANAGEMENT): EdgeType.ADVISORY,
    (OrgCategory.COMMITTEE, OrgCategory.GOVERNANCE): EdgeType.ADVISORY,

    # Department relations
    (OrgCategory.DEPARTMENT, OrgCategory.MANAGEMENT): EdgeType.REPORTS_TO,
    (OrgCategory.DEPARTMENT, OrgCategory.DEPARTMENT): EdgeType.SIBLING,

    # Taskforce relations
    (OrgCategory.TASKFORCE, OrgCategory.MANAGEMENT): EdgeType.REPORTS_TO,
    (OrgCategory.TASKFORCE, OrgCategory.DEPARTMENT): EdgeType.CONTAINS,
    (OrgCategory.TASKFORCE, OrgCategory.TASKFORCE): EdgeType.SIBLING,
}


def infer_edge_type(from_category: OrgCategory, to_category: OrgCategory) -> EdgeType:
    """Infer edge type based on category pair."""
    return EDGE_TYPE_RULES.get((from_category, to_category), EdgeType.UNKNOWN)


# =============================================================================
# Geometric Heuristics with Relative Thresholds
# =============================================================================

@dataclass
class LevelCluster:
    """A cluster of nodes at the same Y level."""
    level: int
    y_center: float
    y_min: float
    y_max: float
    node_ids: list[str]


def get_median_node_height(nodes: list[OrgNode]) -> float:
    """Get median node height for relative threshold calculation."""
    heights = []
    for node in nodes:
        if node.bbox and len(node.bbox) == 4:
            height = node.bbox[3] - node.bbox[1]
            if height > 0:
                heights.append(height)
    return float(np.median(heights)) if heights else 30.0


def cluster_by_y_level(
    nodes: list[OrgNode],
    threshold_factor: float = 0.6,
) -> list[LevelCluster]:
    """
    Cluster nodes by Y coordinate into levels using relative threshold.

    threshold = median_node_height * threshold_factor
    """
    if not nodes:
        return []

    # Calculate relative threshold
    median_height = get_median_node_height(nodes)
    threshold = median_height * threshold_factor

    # Sort by Y center
    sorted_nodes = sorted(nodes, key=lambda n: n.center_y)

    clusters: list[LevelCluster] = []
    current_cluster_nodes: list[OrgNode] = [sorted_nodes[0]]

    for node in sorted_nodes[1:]:
        cluster_y = np.mean([n.center_y for n in current_cluster_nodes])
        if abs(node.center_y - cluster_y) <= threshold:
            current_cluster_nodes.append(node)
        else:
            y_centers = [n.center_y for n in current_cluster_nodes]
            clusters.append(LevelCluster(
                level=len(clusters),
                y_center=float(np.mean(y_centers)),
                y_min=min(y_centers),
                y_max=max(y_centers),
                node_ids=[n.id for n in current_cluster_nodes],
            ))
            current_cluster_nodes = [node]

    # Add last cluster
    if current_cluster_nodes:
        y_centers = [n.center_y for n in current_cluster_nodes]
        clusters.append(LevelCluster(
            level=len(clusters),
            y_center=float(np.mean(y_centers)),
            y_min=min(y_centers),
            y_max=max(y_centers),
            node_ids=[n.id for n in current_cluster_nodes],
        ))

    return clusters


def assign_levels(nodes: list[OrgNode], clusters: list[LevelCluster]) -> None:
    """Assign level to each node based on clustering."""
    node_to_level = {}
    for cluster in clusters:
        for node_id in cluster.node_ids:
            node_to_level[node_id] = cluster.level

    for node in nodes:
        node.level = node_to_level.get(node.id, -1)


def find_parent_candidates(
    node: OrgNode,
    all_nodes: list[OrgNode],
    max_level_diff: int = 3,
    max_candidates: int = 5,
) -> list[ParentCandidate]:
    """
    Find parent candidates for a node with scoring.

    Score = w1 * x_overlap + w2 * (1 - normalized_y_distance) + w3 * center_alignment
    """
    candidates: list[tuple[OrgNode, float]] = []

    # Get page dimensions for normalization
    max_x = max((n.bbox[2] for n in all_nodes if n.bbox), default=1000)
    max_y = max((n.bbox[3] for n in all_nodes if n.bbox), default=1000)

    for other in all_nodes:
        if other.id == node.id:
            continue
        if other.level >= node.level:
            continue

        level_diff = node.level - other.level
        if level_diff > max_level_diff:
            continue

        # Calculate scoring components
        x_overlap = calculate_x_overlap(node.bbox, other.bbox)
        y_distance = abs(node.center_y - other.center_y) / max_y
        center_alignment = 1.0 - min(1.0, abs(node.center_x - other.center_x) / (max_x * 0.3))

        # Weighted score
        w1, w2, w3 = 0.4, 0.3, 0.3
        score = w1 * x_overlap + w2 * (1 - y_distance) + w3 * center_alignment

        # Level bonus (prefer closer levels)
        score *= (1.0 / level_diff)

        candidates.append((other, score))

    # Sort and limit
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [
        ParentCandidate(id=c[0].id, score=c[1])
        for c in candidates[:max_candidates]
    ]


def calculate_x_overlap(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate X-axis overlap ratio between two bboxes."""
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0

    x1_min, _, x1_max, _ = bbox1
    x2_min, _, x2_max, _ = bbox2

    overlap_start = max(x1_min, x2_min)
    overlap_end = min(x1_max, x2_max)

    if overlap_end <= overlap_start:
        return 0.0

    overlap = overlap_end - overlap_start
    min_width = min(x1_max - x1_min, x2_max - x2_min)

    if min_width <= 0:
        return 0.0

    return min(1.0, overlap / min_width)


# =============================================================================
# Cycle Detection (NetworkX)
# =============================================================================

def detect_cycles(nodes: list[OrgNode], edges: list[OrgEdge]) -> list[list[str]]:
    """
    Detect cycles in the graph using NetworkX.

    Returns list of cycles (each cycle is a list of node IDs).
    """
    if not HAS_NETWORKX:
        return []

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        if edge.edge_type in (EdgeType.REPORTS_TO, EdgeType.CONTAINS):
            G.add_edge(edge.from_id, edge.to_id)

    try:
        cycles = list(nx.simple_cycles(G))
        return cycles
    except Exception:
        return []


# =============================================================================
# Main Parser Class
# =============================================================================

class OrgChartParser:
    """
    Parser for organizational charts (B+ approach).

    Features:
    - Multi-stage classification (hint → final)
    - Relative threshold for Y clustering
    - Candidate parent tracking for VLM constraint
    - Cycle detection validation
    """

    def __init__(
        self,
        y_cluster_factor: float = 0.6,
        max_parent_level_diff: int = 3,
        max_parent_candidates: int = 5,
    ):
        self.y_cluster_factor = y_cluster_factor
        self.max_parent_level_diff = max_parent_level_diff
        self.max_parent_candidates = max_parent_candidates

    def parse_from_blocks(
        self,
        blocks: list[dict[str, Any]],
        page_idx: int = 0,
        title: str = "",
        date: str = "",
        vlm_structured_content: str = "",
    ) -> OrgChartGraph:
        """
        Parse org chart from MinerU text blocks.

        When bbox data is unavailable (e.g., from VLM all_text fallback),
        geometric heuristics are skipped. If VLM structured_content with PATH
        notation is available, edges are extracted from it.
        """
        # Step 1: Create nodes from blocks
        nodes, has_bbox = self._create_nodes(blocks, page_idx, title)

        if not nodes:
            return OrgChartGraph(
                title=title,
                date=date,
                page_idx=page_idx,
                needs_review=True,
                review_reasons=["No valid nodes found"],
            )

        # Step 2: Multi-stage classification
        self._classify_nodes(nodes)

        edges: list[OrgEdge] = []

        if has_bbox:
            # Full geometric processing when bbox is available

            # Step 3: Cluster by Y level (relative threshold)
            clusters = cluster_by_y_level(nodes, self.y_cluster_factor)
            assign_levels(nodes, clusters)

            # Step 4: Find parent candidates for each node
            self._find_all_parent_candidates(nodes)

            # Step 5: Infer edges (select best parent)
            edges = self._infer_edges(nodes)
        else:
            # D2 Schema-first: Do NOT parse VLM's Markdown structured_content
            # Reason: VLM's PATH notation / indentation is too fragile and often causes:
            # - Flattening (all units under one parent)
            # - Wrong relationships (hallucinated 監督/諮詢 annotations)
            #
            # Instead, we only do:
            # 1. Text-based classification (classify_by_text rules)
            # 2. Group by category
            # 3. Let validate_and_degrade() handle the output (degrade to groups+summary)
            #
            # Edges will be empty → validation will fail → output degrades to grouped format

            # Assign levels based on category only (no edges to infer from)
            self._assign_levels_by_category(nodes)

        # Step 6: Create groups
        groups = self._create_groups(nodes)

        # Step 7: Generate derived paths
        derived_paths = self._generate_paths(nodes, edges)

        # Step 8: Validate (including cycle detection)
        needs_review, review_reasons = self._validate(nodes, edges, groups)

        # Mark for review if no bbox (limited accuracy)
        if not has_bbox:
            needs_review = True
            review_reasons.append("No bbox data - using VLM text only (limited hierarchy)")

        return OrgChartGraph(
            title=title,
            date=date,
            page_idx=page_idx,
            nodes=nodes,
            edges=edges,
            groups=groups,
            derived_paths=derived_paths,
            needs_review=needs_review,
            review_reasons=review_reasons,
        )

    def _create_nodes(
        self,
        blocks: list[dict[str, Any]],
        page_idx: int,
        title: str = "",
    ) -> tuple[list[OrgNode], bool]:
        """
        Create OrgNode objects from blocks with provenance.

        Returns:
            (nodes, has_bbox): list of nodes and whether bbox data is available
        """
        nodes = []
        has_any_bbox = False

        for i, block in enumerate(blocks):
            text = block.get("text", "").strip()
            raw_bbox = block.get("bbox") or block.get("bbox_norm")

            # Check if we have valid bbox
            has_bbox = raw_bbox is not None and len(raw_bbox) == 4 and any(v != 0 for v in raw_bbox)
            bbox = raw_bbox if has_bbox else [0, 0, 0, 0]

            if has_bbox:
                has_any_bbox = True

            if len(text) < 2:
                continue

            # Apply appropriate noise filter
            if has_bbox:
                if self._is_noise(text, bbox):
                    continue
            else:
                if self._is_vlm_text_noise(text, title):
                    continue

            # Create evidence for provenance
            evidence = Evidence(
                page_idx=page_idx,
                bbox_norm=bbox if has_bbox else None,
                block_id=block.get("block_id", f"b{i:06d}"),
            )

            # Determine source type
            source = SourceType.MINERU_TEXT_BLOCK if has_bbox else SourceType.VLM

            node = OrgNode(
                id=f"n{i:04d}",
                label=text,
                bbox=bbox,
                page_idx=page_idx,
                source=source,
                evidence=evidence,
            )
            nodes.append(node)

        return nodes, has_any_bbox

    def _is_vlm_text_noise(self, text: str, title: str = "") -> bool:
        """
        Check if VLM all_text item is noise (no bbox available).

        Filters out:
        - Page numbers (壹-4, 頁1, etc.)
        - Title text
        - Parenthetical annotations
        - Date-only strings
        """
        text_clean = text.strip()

        # Skip empty or very short
        if len(text_clean) < 2:
            return True

        # Skip if text is just parenthetical annotation
        if text_clean.startswith("(") and text_clean.endswith(")"):
            return True

        # Skip page numbers (Chinese or Arabic)
        # Patterns: 壹-4, 頁1, Page 1, 1/10, etc.
        page_num_patterns = [
            r"^[壹貳參肆伍陸柒捌玖拾]+[-－]\d+$",  # 壹-4
            r"^頁?\s*\d+$",  # 頁1
            r"^Page\s*\d+$",  # Page 1
            r"^\d+\s*/\s*\d+$",  # 1/10
            r"^[a-zA-Z]?\d{1,3}$",  # P1, 1, etc. (short alphanumeric)
        ]
        for pattern in page_num_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True

        # Skip if text contains title (likely header/watermark)
        if title and len(text_clean) > 10:
            # Normalize parentheses for comparison
            def normalize_parens(s: str) -> str:
                return s.replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace(" ", "")

            title_norm = normalize_parens(title)
            text_norm = normalize_parens(text_clean)

            # Check if this text contains significant portion of title
            if title_norm in text_norm or text_norm in title_norm:
                return True

        # Skip date-only strings (114.06.20, 2024/01/01, etc.)
        date_patterns = [
            r"^\d{2,4}[./-]\d{1,2}[./-]\d{1,2}$",  # 114.06.20, 2024/01/01
            r"^\d{2,4}年\d{1,2}月\d{1,2}日$",  # 114年06月20日
        ]
        for pattern in date_patterns:
            if re.match(pattern, text_clean):
                return True

        return False

    def _is_noise(self, text: str, bbox: list[float]) -> bool:
        """
        Check if text is noise.

        Handles both normalized (0-1) and pixel-based bbox coordinates.
        """
        if text.replace("-", "").replace(".", "").isdigit():
            return True

        # Short text in corner regions (page numbers, headers/footers)
        if len(text) <= 5 and bbox and len(bbox) == 4:
            x0, y0, x1, y1 = bbox

            # Detect if bbox is normalized (0-1 range) or pixel-based
            is_normalized = max(x0, y0, x1, y1) <= 1.0

            if is_normalized:
                # Normalized: check corners (top/bottom 5%, left/right 10%)
                in_corner_y = y0 < 0.05 or y1 > 0.95
                in_corner_x = x0 < 0.1 or x1 > 0.9
            else:
                # Pixel-based: assume 1000x1000 coordinate system
                in_corner_y = y0 < 50 or y1 > 950
                in_corner_x = x0 < 100 or x1 > 900

            if in_corner_y and in_corner_x:
                return True

        return False

    def _classify_nodes(self, nodes: list[OrgNode]) -> None:
        """Multi-stage classification: hint → final."""
        for node in nodes:
            hint, final, confidence = classify_by_text(node.label)
            node.category_hint = hint
            node.category = final
            node.hint_confidence = confidence
            node.confidence = confidence

    def _assign_levels_by_category(self, nodes: list[OrgNode]) -> None:
        """
        Assign levels based on category when bbox is not available.

        Level mapping:
        - 0: GOVERNANCE (董事會, 監察人, 稽核室)
        - 1: MANAGEMENT (院長, 副院長, 主任秘書)
        - 2: COMMITTEE (各委員會)
        - 3: DEPARTMENT (研究所, 處, 室)
        - 4: TASKFORCE (中心, 辦公室)
        - 5: UNKNOWN
        """
        category_to_level = {
            OrgCategory.GOVERNANCE: 0,
            OrgCategory.MANAGEMENT: 1,
            OrgCategory.COMMITTEE: 2,
            OrgCategory.DEPARTMENT: 3,
            OrgCategory.TASKFORCE: 4,
            OrgCategory.UNKNOWN: 5,
        }
        for node in nodes:
            # Use hint if final is UNKNOWN
            category = node.category if node.category != OrgCategory.UNKNOWN else node.category_hint
            node.level = category_to_level.get(category, 5)

    def _parse_vlm_paths(
        self,
        vlm_content: str,
        nodes: list[OrgNode],
    ) -> list[OrgEdge]:
        """
        Parse VLM's structured content to extract edges.

        Supports two formats:
        1. PATH notation: "董事會 > 院長 > 研究一所"
        2. Markdown indentation: "- 董事會\\n    - 院長\\n        - 研究一所"
        """
        edges: list[OrgEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        # Build label -> node mapping (for fuzzy matching)
        label_to_node: dict[str, OrgNode] = {}
        for node in nodes:
            # Multiple matching keys for flexibility
            label_clean = node.label.split("(")[0].split("（")[0].strip()
            label_to_node[label_clean] = node
            label_to_node[node.label] = node
            # Also try without spaces
            label_to_node[label_clean.replace(" ", "")] = node

        def add_edge(child_label: str, parent_label: str) -> bool:
            """Add an edge if both nodes exist. Returns True if added."""
            # Clean labels
            child_clean = child_label.replace("**", "").strip()
            parent_clean = parent_label.replace("**", "").strip()

            # Try exact match first, then normalized
            child_node = (
                label_to_node.get(child_clean) or
                label_to_node.get(child_clean.split("(")[0].split("（")[0].strip()) or
                label_to_node.get(child_clean.replace(" ", ""))
            )
            parent_node = (
                label_to_node.get(parent_clean) or
                label_to_node.get(parent_clean.split("(")[0].split("（")[0].strip()) or
                label_to_node.get(parent_clean.replace(" ", ""))
            )

            if not child_node or not parent_node:
                return False
            if child_node.id == parent_node.id:
                return False

            edge_key = (child_node.id, parent_node.id)
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)

            edge_type = infer_edge_type(child_node.category, parent_node.category)

            edge = OrgEdge(
                from_id=child_node.id,
                to_id=parent_node.id,
                edge_type=edge_type,
                confidence=0.8,
                source=SourceType.VLM,
                decision="vlm_structured",
            )
            edges.append(edge)

            child_node.chosen_parent = parent_node.id
            child_node.parent_decision = "vlm"
            return True

        # === Strategy 1: Parse PATH notation (A > B > C) ===
        for line in vlm_content.split("\n"):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue

            if " > " in line_stripped:
                # Remove list markers and parenthetical descriptions
                clean_line = line_stripped.lstrip("-* ")
                if "(" in clean_line:
                    clean_line = clean_line.split("(")[0].strip()
                if "（" in clean_line:
                    clean_line = clean_line.split("（")[0].strip()

                parts = [p.strip() for p in clean_line.split(" > ")]
                for i in range(len(parts) - 1):
                    add_edge(parts[i + 1], parts[i])

        # === Strategy 2: Parse Markdown indentation ===
        # If we found edges from PATH notation, skip indentation parsing
        if not edges:
            self._parse_markdown_indentation(vlm_content, label_to_node, add_edge)

        return edges

    def _parse_markdown_indentation(
        self,
        vlm_content: str,
        label_to_node: dict[str, "OrgNode"],
        add_edge_fn,
    ) -> None:
        """
        Parse markdown list indentation to extract hierarchy.

        Supports:
        1. Indentation-based hierarchy:
           - Level 0 item
               - Level 1 item (4 spaces)
        2. Explicit relationship annotations:
           - 監察人（監督：董事會）-> oversight edge
           - 人事評議委員會（諮詢：院長）-> advisory edge
        """
        # Stack to track parent at each indent level: [(indent, label), ...]
        parent_stack: list[tuple[int, str]] = []

        for line in vlm_content.split("\n"):
            # Skip empty lines and headers
            if not line.strip() or line.strip().startswith("#"):
                # Reset stack on section headers
                if line.strip().startswith("#"):
                    parent_stack = []
                continue

            # Calculate indent level (count leading spaces)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Check if it's a list item
            if not (stripped.startswith("- ") or stripped.startswith("* ")):
                continue

            # Extract full item text (remove list marker)
            item_text = stripped[2:].strip()

            # === Parse explicit relationship annotations ===
            # Format: "單位名稱（監督：目標）" or "單位名稱（諮詢：目標）"
            relation_match = re.search(r'[（(](監督|諮詢|隸屬)[：:]\s*([^）)]+)[）)]', item_text)
            if relation_match:
                relation_type = relation_match.group(1)
                target_label = relation_match.group(2).strip()
                source_label = item_text.split("（")[0].split("(")[0].strip().replace("**", "")

                # Map relation type to edge creation
                if relation_type == "監督":
                    # Oversight: source oversights target
                    add_edge_fn(source_label, target_label)
                elif relation_type == "諮詢":
                    # Advisory: source advises target
                    add_edge_fn(source_label, target_label)
                elif relation_type == "隸屬":
                    # Reports to
                    add_edge_fn(source_label, target_label)
                continue  # Don't add to parent stack for explicit relations

            # === Standard indentation-based hierarchy ===
            # Extract label (remove markdown bold and parenthetical descriptions)
            label = item_text.replace("**", "")
            if "(" in label:
                label = label.split("(")[0].strip()
            if "（" in label:
                label = label.split("（")[0].strip()

            if not label or len(label) < 2:
                continue

            # Pop parents with indent >= current (they're not our parent)
            while parent_stack and parent_stack[-1][0] >= indent:
                parent_stack.pop()

            # If we have a parent, create edge
            if parent_stack:
                parent_label = parent_stack[-1][1]
                add_edge_fn(label, parent_label)

            # Push current item as potential parent for next items
            parent_stack.append((indent, label))

    def _assign_levels_from_edges(
        self,
        nodes: list[OrgNode],
        edges: list[OrgEdge],
    ) -> None:
        """
        Assign levels based on edge hierarchy (BFS from roots).

        Only REPORTS_TO/CONTAINS are used for level calculation.
        OVERSIGHT/ADVISORY are non-hierarchical relationships.
        """
        # Build adjacency: parent -> children (only hierarchical edges)
        children_map: dict[str, list[str]] = {}
        has_parent: set[str] = set()

        for edge in edges:
            if edge.edge_type in (EdgeType.REPORTS_TO, EdgeType.CONTAINS):
                parent_id = edge.to_id
                child_id = edge.from_id
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(child_id)
                has_parent.add(child_id)

        # Find roots (nodes with no parent)
        node_ids = {n.id for n in nodes}
        root_ids = node_ids - has_parent

        # BFS to assign levels
        node_map = {n.id: n for n in nodes}
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(rid, 0) for rid in root_ids]

        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            if node_id in node_map:
                node_map[node_id].level = level

            for child_id in children_map.get(node_id, []):
                if child_id not in visited:
                    queue.append((child_id, level + 1))

        # Assign level 99 to orphan nodes
        for node in nodes:
            if node.id not in visited:
                node.level = 99

    def _find_all_parent_candidates(self, nodes: list[OrgNode]) -> None:
        """Find parent candidates for all nodes."""
        for node in nodes:
            candidates = find_parent_candidates(
                node,
                nodes,
                self.max_parent_level_diff,
                self.max_parent_candidates,
            )
            node.candidate_parents = candidates

    def generate_edge_candidates(self, nodes: list[OrgNode]) -> list[EdgeCandidate]:
        """
        D3.1: Generate edge candidates from bbox heuristics.

        For each non-root node, generate top-N parent candidates
        based on geometric proximity. These candidates will be
        presented to VLM#2 for selection.

        Returns:
            List of EdgeCandidate objects for VLM selection
        """
        candidates: list[EdgeCandidate] = []
        node_map = {n.id: n for n in nodes}

        for node in nodes:
            # Skip if no parent candidates or is root
            if not node.candidate_parents:
                continue

            # Convert ParentCandidate to (id, label, score) tuples
            parent_tuples: list[tuple[str, str, float]] = []
            for pc in node.candidate_parents:
                parent_node = node_map.get(pc.id)
                if parent_node:
                    parent_tuples.append((pc.id, parent_node.label, pc.score))

            if parent_tuples:
                candidates.append(EdgeCandidate(
                    child_id=node.id,
                    child_label=node.label,
                    parent_candidates=parent_tuples,
                ))

        return candidates

    def parse_vlm2_edges(
        self,
        vlm_edges_output: dict[str, Any],
        nodes: list[OrgNode],
        candidates: list[EdgeCandidate],
    ) -> list[OrgEdge]:
        """
        D3.3: Parse VLM#2 edge selection response into OrgEdge objects.

        Args:
            vlm_edges_output: Parsed JSON from VLM#2 (OrgChartEdgesOutput format)
            nodes: List of OrgNode objects
            candidates: Original edge candidates (for ID lookup)

        Returns:
            List of OrgEdge objects with VLM-selected parents
        """
        edges: list[OrgEdge] = []

        # Build label -> node mapping
        label_to_node: dict[str, OrgNode] = {}
        for node in nodes:
            label_clean = node.label.split("(")[0].split("（")[0].strip()
            label_to_node[label_clean] = node
            label_to_node[node.label] = node

        # Build child_label -> EdgeCandidate mapping (for parent ID lookup)
        child_to_candidates: dict[str, EdgeCandidate] = {}
        for ec in candidates:
            child_to_candidates[ec.child_label] = ec

        # Edge type string to EdgeType enum mapping
        edge_type_map = {
            "reports_to": EdgeType.REPORTS_TO,
            "oversight": EdgeType.OVERSIGHT,
            "advisory": EdgeType.ADVISORY,
            "contains": EdgeType.CONTAINS,
            "sibling": EdgeType.SIBLING,
            "root": EdgeType.UNKNOWN,  # Root nodes don't have edges
        }

        vlm_edges = vlm_edges_output.get("edges", [])
        for vlm_edge in vlm_edges:
            child_label = vlm_edge.get("child", "")
            parent_label = vlm_edge.get("parent")
            edge_type_str = vlm_edge.get("edge_type", "reports_to")
            confidence = vlm_edge.get("confidence", 0.7)

            # Skip root nodes (no edge needed)
            if parent_label is None or edge_type_str == "root":
                continue

            # Find child node
            child_node = (
                label_to_node.get(child_label) or
                label_to_node.get(child_label.split("(")[0].strip())
            )
            if not child_node:
                continue

            # Find parent node - first try direct lookup, then check candidates
            parent_node = (
                label_to_node.get(parent_label) or
                label_to_node.get(parent_label.split("(")[0].strip())
            )

            # If not found directly, search in candidate list
            if not parent_node:
                ec = child_to_candidates.get(child_label)
                if ec:
                    for pid, plabel, _ in ec.parent_candidates:
                        if plabel == parent_label or plabel.startswith(parent_label):
                            parent_node = next((n for n in nodes if n.id == pid), None)
                            break

            if not parent_node:
                continue

            # Create edge
            edge_type = edge_type_map.get(edge_type_str, EdgeType.UNKNOWN)
            edge = OrgEdge(
                from_id=child_node.id,
                to_id=parent_node.id,
                edge_type=edge_type,
                confidence=confidence,
                needs_review=confidence < 0.7,
                source=SourceType.VLM,
                decision="vlm2_selected",
            )
            edges.append(edge)

            # Update node's chosen parent
            child_node.chosen_parent = parent_node.id
            child_node.parent_decision = "vlm2"

        return edges

    def _infer_edges(self, nodes: list[OrgNode]) -> list[OrgEdge]:
        """Infer edges by selecting best parent from candidates."""
        edges = []

        for node in nodes:
            if not node.candidate_parents:
                continue

            # Select best parent (highest score)
            best = node.candidate_parents[0]
            node.chosen_parent = best.id
            node.parent_decision = "heuristic"

            # Find parent node for edge type inference
            parent_node = next((n for n in nodes if n.id == best.id), None)
            if not parent_node:
                continue

            edge_type = infer_edge_type(node.category, parent_node.category)

            edge = OrgEdge(
                from_id=node.id,
                to_id=best.id,
                edge_type=edge_type,
                confidence=best.score,
                needs_review=best.score < 0.5 or edge_type == EdgeType.UNKNOWN,
                source=SourceType.HEURISTIC,
                decision="heuristic",
            )
            edges.append(edge)

        # Add sibling edges
        edges.extend(self._add_sibling_edges(nodes, edges))

        return edges

    def _add_sibling_edges(
        self,
        nodes: list[OrgNode],
        existing_edges: list[OrgEdge],
    ) -> list[OrgEdge]:
        """Add sibling edges for nodes at the same level with same parent."""
        sibling_edges = []

        parent_to_children: dict[str, list[OrgNode]] = {}
        for edge in existing_edges:
            if edge.edge_type == EdgeType.REPORTS_TO:
                parent_id = edge.to_id
                child_node = next((n for n in nodes if n.id == edge.from_id), None)
                if child_node:
                    if parent_id not in parent_to_children:
                        parent_to_children[parent_id] = []
                    parent_to_children[parent_id].append(child_node)

        for children in parent_to_children.values():
            if len(children) < 2:
                continue

            level_groups: dict[int, list[OrgNode]] = {}
            for child in children:
                if child.level not in level_groups:
                    level_groups[child.level] = []
                level_groups[child.level].append(child)

            for level_nodes in level_groups.values():
                if len(level_nodes) < 2:
                    continue

                sorted_nodes = sorted(level_nodes, key=lambda n: n.center_x)
                for i in range(len(sorted_nodes) - 1):
                    sibling_edges.append(OrgEdge(
                        from_id=sorted_nodes[i].id,
                        to_id=sorted_nodes[i + 1].id,
                        edge_type=EdgeType.SIBLING,
                        confidence=0.8,
                        needs_review=False,
                        source=SourceType.HEURISTIC,
                        decision="heuristic",
                    ))

        return sibling_edges

    def _create_groups(self, nodes: list[OrgNode]) -> list[OrgGroup]:
        """Create functional groups from classified nodes."""
        groups: dict[OrgCategory, list[str]] = {}

        for node in nodes:
            cat = node.category if node.category != OrgCategory.UNKNOWN else node.category_hint
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(node.id)

        descriptions = {
            OrgCategory.GOVERNANCE: "最高權力與監督機構",
            OrgCategory.MANAGEMENT: "經營管理層級",
            OrgCategory.COMMITTEE: "跨部門委員會，負責特定院務",
            OrgCategory.DEPARTMENT: "正式編制單位，平行一級單位",
            OrgCategory.TASKFORCE: "任務編組，專案性質",
            OrgCategory.UNKNOWN: "待分類單位",
        }

        result = []
        for category in OrgCategory:
            if category in groups and groups[category]:
                result.append(OrgGroup(
                    name=category.value,
                    members=groups[category],
                    confidence=min(
                        next((n.confidence for n in nodes if n.id == nid), 0.5)
                        for nid in groups[category]
                    ),
                    description=descriptions.get(category, ""),
                ))

        return result

    def _generate_paths(
        self,
        nodes: list[OrgNode],
        edges: list[OrgEdge],
    ) -> list[str]:
        """Generate PATH notation from graph."""
        paths = []
        node_map = {n.id: n for n in nodes}
        parent_map: dict[str, str] = {}

        for edge in edges:
            if edge.edge_type == EdgeType.REPORTS_TO:
                parent_map[edge.from_id] = edge.to_id

        def get_path(node_id: str) -> list[str]:
            path = []
            current = node_id
            visited = set()
            while current and current not in visited:
                visited.add(current)
                node = node_map.get(current)
                if node:
                    path.append(node.label)
                current = parent_map.get(current)
            return list(reversed(path))

        for node in nodes:
            path = get_path(node.id)
            if path:
                path_str = " > ".join(path)
                if path_str not in paths:
                    paths.append(path_str)

        paths.sort(key=lambda p: (p.count(">"), p))
        return paths

    def _validate(
        self,
        nodes: list[OrgNode],
        edges: list[OrgEdge],
        groups: list[OrgGroup],
    ) -> tuple[bool, list[str]]:
        """Validate the graph with cycle detection."""
        reasons = []

        # Check 1: Cycle detection
        cycles = detect_cycles(nodes, edges)
        if cycles:
            reasons.append(f"Cycles detected: {len(cycles)} cycles")

        # Check 2: Unknown edge ratio
        unknown_edges = [e for e in edges if e.edge_type == EdgeType.UNKNOWN]
        if edges and len(unknown_edges) / len(edges) > 0.3:
            reasons.append(f"High unknown edge ratio: {len(unknown_edges)}/{len(edges)}")

        # Check 3: Multiple roots
        reports_to_edges = [e for e in edges if e.edge_type == EdgeType.REPORTS_TO]
        child_ids = {e.from_id for e in reports_to_edges}
        root_nodes = [n for n in nodes if n.id not in child_ids]
        if len(root_nodes) > 3:
            reasons.append(f"Multiple root nodes: {len(root_nodes)}")

        # Check 4: Flattening detection
        parent_counts: dict[str, int] = {}
        for edge in reports_to_edges:
            parent_counts[edge.to_id] = parent_counts.get(edge.to_id, 0) + 1

        if parent_counts:
            max_children = max(parent_counts.values())
            total_children = len(child_ids)
            if total_children > 5 and max_children / total_children > 0.8:
                reasons.append(f"Potential flattening: one parent has {max_children}/{total_children} children")

        # Check 5: Edge coverage (orphan ratio)
        if nodes:
            orphan_count = len(nodes) - len(child_ids) - len(root_nodes)
            orphan_ratio = orphan_count / len(nodes) if len(nodes) > 0 else 0
            if orphan_ratio > 0.2:
                reasons.append(f"High orphan ratio: {orphan_count}/{len(nodes)} ({orphan_ratio:.1%})")

        # Check 6: Unknown category ratio
        unknown_nodes = [n for n in nodes if n.category == OrgCategory.UNKNOWN]
        if nodes and len(unknown_nodes) / len(nodes) > 0.4:
            reasons.append(f"High unknown category ratio: {len(unknown_nodes)}/{len(nodes)}")

        return bool(reasons), reasons


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_org_chart(
    blocks: list[dict[str, Any]],
    page_idx: int = 0,
    title: str = "",
    date: str = "",
) -> OrgChartGraph:
    """Convenience function to parse org chart from blocks."""
    parser = OrgChartParser()
    return parser.parse_from_blocks(blocks, page_idx, title, date)
