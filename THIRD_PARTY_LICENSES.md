# Third-Party Licenses

This project is intended to be released under the Apache License 2.0 for the project's own source code. Third-party libraries, parsers, model weights, model servers, API providers, and generated assets remain under their own licenses and terms.

This file is a release checklist and licensing boundary, not legal advice. Before publishing a tagged release, verify the exact installed package versions, model cards, and upstream license files.

## Project Code

- Scope: source code and documentation authored for this repository.
- Intended license: Apache License 2.0.
- License file: `LICENSE`.
- Does not cover: third-party dependencies, model weights, hosted API terms, user-uploaded documents, generated parsing outputs, or local runtime caches.

## Critical Runtime Components

| Component | Current role | License status to treat as | Release decision |
| --- | --- | --- | --- |
| PyMuPDF / MuPDF | Core PDF/image document handling via `PyMuPDF>=1.27.2.3` | AGPL-3.0 or commercial license, based on upstream PyMuPDF/Artifex licensing | Keep documented as a copyleft/commercial dependency. If a fully permissive distribution is required, make this backend optional or replace it. |
| MinerU | Optional parser extra via `mineru>=3.1.15` | MinerU Open Source License for current 3.1.x line; older versions may have been AGPLv3 | Keep optional. Users must verify the exact MinerU version and comply with MinerU's own license and model terms. |
| App-level VLM | Optional enrichment for forms, figures, diagrams, and tables | Depends on the configured local model or remote API provider | Do not bundle model weights or API keys. Document that users choose and license their own model/provider. |

## VLM Boundary

The application supports two VLM deployment styles:

- Local model server: users run a local OpenAI-compatible server, such as Ollama, vLLM, SGLang, or LMDeploy. The selected model weights and runtime keep their own licenses.
- Remote API endpoint: users configure an OpenAI-compatible API base URL and API key. Documents and extracted images may be sent to that configured provider when VLM enrichment is enabled.

No default project configuration should send documents to a cloud model. Published examples must use placeholders for API keys and endpoints.

## Model Weights

Do not commit or redistribute model weights in this repository unless their license has been reviewed separately. For any recommended model, document:

- Exact model name and version.
- Source URL or model card.
- License identifier or license file.
- Commercial-use status.
- Attribution or notice requirements.
- Any usage restrictions, including hosted-service, scale, geography, or field of use limits.

Some Qwen repositories and model releases use Apache License 2.0, but the license must be checked per exact model and model card. Do not describe all Qwen or VLM models as Apache-2.0 by default.

## Backend Python Dependencies

Primary backend dependencies are declared in `backend/pyproject.toml`. Before release:

- Generate a dependency/license report from the resolved environment.
- Review transitive dependencies, not only direct dependencies.
- Pay special attention to copyleft, non-commercial, source-available, model, dataset, OCR, PDF, and native binary dependencies.
- Keep `mineru` in optional dependencies unless the release intentionally accepts MinerU's license obligations as part of the default runtime.

## Frontend npm Dependencies

Primary frontend dependencies are declared in `frontend/package.json` and resolved in `frontend/package-lock.json`. Before release:

- Generate a dependency/license report from the lockfile.
- Review transitive dependencies.
- Keep generated `frontend/dist/` and `frontend/node_modules/` out of source releases unless the release process intentionally includes built artifacts.

## Runtime Data Not Covered

The Apache-2.0 project license does not grant rights to publish or redistribute:

- Documents uploaded by users.
- Generated extraction outputs derived from private/user documents.
- Benchmark corpora that are not synthetic or public-domain/publicly licensed.
- `.env` files, credentials, internal endpoints, and deployment-specific config.
- Model caches, OCR caches, parser workspaces, or local databases.

## Release Gate

Before publishing to GitHub:

- Confirm the repository root is `doc1/`, not the outer workspace.
- Confirm `backend/.env`, `frontend/.env`, `backend/workspace/`, `frontend/node_modules/`, `frontend/dist/`, and local model caches are not committed.
- Run backend tests and lint.
- Run frontend lint and build.
- Verify the API is local/trusted-network by default and has no public-facing auth claims.
- Re-check PyMuPDF, MinerU, and any recommended VLM model licenses against the exact versions being documented.
