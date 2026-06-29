"""
Application configuration and pipeline profiles.

Profiles: FAST, ACCURATE
"""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def semark_env(name: str) -> AliasChoices:
    """Prefer SEMARK_* settings while accepting legacy DOC_PARSER_* names."""

    return AliasChoices(f"SEMARK_{name}", f"DOC_PARSER_{name}")


class MinerUMethod(StrEnum):
    AUTO = "auto"
    TXT = "txt"
    OCR = "ocr"


class MinerUBackend(StrEnum):
    PIPELINE = "pipeline"
    HYBRID_AUTO_ENGINE = "hybrid-auto-engine"
    HYBRID_HTTP_CLIENT = "hybrid-http-client"
    VLM_AUTO_ENGINE = "vlm-auto-engine"
    VLM_HTTP_CLIENT = "vlm-http-client"


class SemanticOutputLanguage(StrEnum):
    """Language for generated semantic/RAG scaffolding."""

    AUTO = "auto"
    ZH_TW = "zh-TW"
    EN = "en"


class MinerUConfig(BaseModel):
    """MinerU CLI configuration - maps to CLI arguments."""

    method: MinerUMethod = MinerUMethod.AUTO
    backend: MinerUBackend = MinerUBackend.HYBRID_AUTO_ENGINE
    lang: str = "chinese_cht"
    table: bool = True
    formula: bool = True
    start_page: int | None = None
    end_page: int | None = None
    api_url: str | None = None  # Existing mineru-api / mineru-router URL
    vlm_url: str | None = None  # OpenAI-compatible URL for *-http-client backends
    vlm_model_name: str | None = None
    vlm_api_key: str | None = None
    model_source: str | None = None  # huggingface | modelscope | local

    # Environment variables for MinerU
    pdf_render_timeout: int = 300
    pdf_render_threads: int | None = 4
    table_merge_enable: bool = True
    processing_window_size: int | None = None
    api_max_concurrent_requests: int | None = None
    local_api_startup_timeout_seconds: int | None = 300
    task_result_timeout_seconds: int | None = 3600
    task_result_download_timeout_seconds: int | None = 600
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None


class HTMLExtractor(StrEnum):
    MAGIC_HTML = "magic-html"
    DRIPPER = "dripper"


class HTMLConfig(BaseModel):
    """HTML parser configuration."""

    extractor: HTMLExtractor = HTMLExtractor.MAGIC_HTML
    dripper_endpoint: str | None = None  # FastAPI server URL for dripper


class VLMDecodeParams(BaseModel):
    """VLM decoding parameters."""

    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int | None = None
    max_tokens: int = Field(default=1024, ge=1, le=131072)  # Up to 128K for modern VLMs
    repetition_penalty: float = Field(default=1.0, ge=1, le=2)


class VLMApiMode(StrEnum):
    """VLM API mode."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    LMDEPLOY = "lmdeploy"
    OPENAI = "openai"


class VLMImageMode(StrEnum):
    """How to send images to VLM."""

    DATA_URI = "data_uri"  # Base64 encoded in message
    STATIC_URL = "static_url"  # URL to static file server


class VLMConfig(BaseModel):
    """VLM adapter configuration - OpenAI-compatible interface."""

    base_url: str = "http://localhost:11434/v1"  # Ollama default
    api_key: str = "ollama"  # Ollama doesn't require key
    model: str = "qwen2.5-vl:7b"
    api_mode: VLMApiMode = VLMApiMode.OLLAMA
    decode_params: VLMDecodeParams = Field(default_factory=VLMDecodeParams)
    request_timeout_seconds: float = 180.0

    # vLLM specific
    chat_template: str | None = None  # Path or name for vLLM chat template

    # Image transfer
    image_mode: VLMImageMode = VLMImageMode.STATIC_URL  # Prefer static URL
    static_url_base: str = "http://localhost:8585/api/assets"  # Backend asset URL

    # Vision options
    crop_padding: int = 10  # pixels
    include_page_thumbnail: bool = False
    form_mode: Literal["page", "block"] = "page"

    # JSON schema strict mode (Ollama/vLLM structured outputs)
    # When True, uses response_format with JSON schema to constrain output
    use_json_schema: bool = False  # Default off (needs model support)

    # Capability probe cache
    probe_result: dict[str, Any] | None = None
    probe_timestamp: str | None = None


class EnrichConfig(BaseModel):
    """Enrichment stage configuration."""

    enable_vlm: bool = True  # Master switch for VLM enrichment

    # VLM enrichment switches (only effective when enable_vlm=True)
    vlm_enrich_forms: bool = True  # Form field extraction + filling guide
    vlm_enrich_figures: bool = True  # Figure/diagram captioning
    vlm_enrich_tables: bool = False  # Table summarization (expensive)

    # Gating heuristics for forms
    form_filename_patterns: list[str] = Field(
        default_factory=lambda: [
            "申請", "申請書", "申請表", "表單", "請假", "加班", "進修", "附件",
            "form", "application", "authorization", "authorisation", "consent", "request",
            "claim", "transcript", "tax", "irs", "ssa",
        ]
    )
    min_text_ratio_for_vlm: float = 0.3  # pages with less text are candidates

    # VLM gating for tables (only when vlm_enrich_tables=True)
    table_vlm_budget: int = 10  # Max tables to process per document (0=unlimited)
    table_min_cells: int = 4  # Skip tiny tables (rows*cols < this)
    table_max_cells: int = 200  # Large tables will be truncated (not skipped)

    # Table truncation settings (for large tables exceeding max_cells)
    table_truncate_head_rows: int = 10  # Keep first N data rows
    table_truncate_tail_rows: int = 5  # Keep last N data rows

    # Layout table detection thresholds
    table_layout_min_ratio: float = 0.3  # Tables with lower non_empty_ratio are considered layout
    table_layout_min_chars_per_cell: float = 2.0  # Tables with fewer chars per cell are layout


class PackageConfig(BaseModel):
    """Package stage configuration."""

    generate_dataset_md: bool = True
    generate_rag_md: bool = True
    generate_chunks: bool = True
    semantic_output_language: SemanticOutputLanguage = SemanticOutputLanguage.AUTO
    enable_semantic_repair: bool = True

    # Chunk settings
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    mineru: MinerUConfig = Field(default_factory=MinerUConfig)
    html: HTMLConfig = Field(default_factory=HTMLConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    review_vlm: VLMConfig = Field(default_factory=VLMConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)


class ProfileName(StrEnum):
    FAST = "fast"
    ACCURATE = "accurate"


# Built-in profiles
PROFILES: dict[ProfileName, PipelineConfig] = {
    ProfileName.FAST: PipelineConfig(
        mineru=MinerUConfig(
            method=MinerUMethod.AUTO,
            backend=MinerUBackend.PIPELINE,
            table=True,
            formula=False,
        ),
        enrich=EnrichConfig(
            enable_vlm=False,  # Skip VLM in fast mode
            vlm_enrich_forms=True,  # Would run if enable_vlm was True
            vlm_enrich_figures=False,
            vlm_enrich_tables=False,
        ),
    ),
    ProfileName.ACCURATE: PipelineConfig(
        mineru=MinerUConfig(
            method=MinerUMethod.OCR,
            backend=MinerUBackend.PIPELINE,
            table=True,
            formula=True,
        ),
        enrich=EnrichConfig(
            enable_vlm=True,
            vlm_enrich_forms=True,
            vlm_enrich_figures=True,
            vlm_enrich_tables=False,
            # Keep table VLM off by default; deterministic row-level serialization is faster and more stable.
            table_vlm_budget=5,
            table_max_cells=300,  # Allow larger tables with truncation
            table_truncate_head_rows=15,  # Keep more context
            table_truncate_tail_rows=10,
        ),
    ),
}


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    # Paths
    workspace_path: Path = Field(default=Path("workspace"), validation_alias=semark_env("WORKSPACE_PATH"))
    database_path: Path = Field(default=Path("workspace/doc_parser.db"), validation_alias=semark_env("DATABASE_PATH"))

    # Server
    host: str = Field(default="127.0.0.1", validation_alias=semark_env("HOST"))
    port: int = Field(default=8000, validation_alias=semark_env("PORT"))
    debug: bool = Field(default=False, validation_alias=semark_env("DEBUG"))
    enable_local_path_ingest: bool = Field(default=False, validation_alias=semark_env("ENABLE_LOCAL_PATH_INGEST"))
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5070",
            "http://127.0.0.1:5070",
        ],
        validation_alias=semark_env("CORS_ORIGINS"),
    )
    cors_allow_private_lan: bool = Field(default=False, validation_alias=semark_env("CORS_ALLOW_PRIVATE_LAN"))

    # Default profile
    default_profile: ProfileName = Field(default=ProfileName.ACCURATE, validation_alias=semark_env("DEFAULT_PROFILE"))

    # MinerU
    mineru_cli_path: str = Field(default="mineru", validation_alias=semark_env("MINERU_CLI_PATH"))  # Assumes in PATH
    mineru_model_source: str = Field(default="huggingface", validation_alias=semark_env("MINERU_MODEL_SOURCE"))
    mineru_api_url: str | None = Field(default=None, validation_alias=semark_env("MINERU_API_URL"))
    mineru_vlm_url: str | None = Field(default=None, validation_alias=semark_env("MINERU_VLM_URL"))
    mineru_vlm_model_name: str | None = Field(default=None, validation_alias=semark_env("MINERU_VLM_MODEL_NAME"))
    mineru_vlm_api_key: str | None = Field(default=None, validation_alias=semark_env("MINERU_VLM_API_KEY"))

    # VLM defaults (can be overridden per-run)
    vlm_base_url: str = Field(default="http://localhost:11434/v1", validation_alias=semark_env("VLM_BASE_URL"))
    vlm_api_key: str = Field(default="ollama", validation_alias=semark_env("VLM_API_KEY"))
    vlm_model: str = Field(default="qwen2.5-vl:7b", validation_alias=semark_env("VLM_MODEL"))

    # Reviewer VLM defaults. Leave unset to reuse the enrichment VLM.
    review_vlm_base_url: str | None = Field(default=None, validation_alias=semark_env("REVIEW_VLM_BASE_URL"))
    review_vlm_api_key: str | None = Field(default=None, validation_alias=semark_env("REVIEW_VLM_API_KEY"))
    review_vlm_model: str | None = Field(default=None, validation_alias=semark_env("REVIEW_VLM_MODEL"))

    @property
    def store_path(self) -> Path:
        return self.workspace_path / "store"

    @property
    def docs_path(self) -> Path:
        return self.store_path / "docs"

    def get_doc_path(self, doc_id: str) -> Path:
        return self.docs_path / doc_id

    def get_run_path(self, doc_id: str, run_id: str) -> Path:
        return self.get_doc_path(doc_id) / "runs" / run_id


# Global settings instance
settings = Settings()
