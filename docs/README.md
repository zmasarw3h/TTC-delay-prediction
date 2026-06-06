# Documentation Index

## Primary Docs

- [Model card](model_card.md): intended use, data, leakage controls, metrics, calibration, and limitations.
- [Technical report](technical_report.md): concise end-to-end project narrative and modeling decisions.
- [Architecture](architecture.md): pipeline diagram and component descriptions.
- [API service](api_service.md): FastAPI endpoints and local frontend behavior.
- [API input contract](api_input_contract.md): accepted fields, normalization, leakage rejection, and historical lookup defaults.
- [Historical feature lookup](historical_feature_lookup.md): prior-only inference-time historical feature computation.
- [Feature engineering](feature_engineering.md): time features, v1/v2 historical features, and split generation.
- [Categorical normalization](categorical_normalization.md): deterministic normalization contract for key categorical fields.
- [Final QA checklist](final_qa_checklist.md): release-readiness checks before making the repo public.

## Archived Development Notes

Older planning, phase-specific, and exploratory audit notes live in [archive/](archive/). They are kept for transparency but are not the recommended starting point for reviewers.
