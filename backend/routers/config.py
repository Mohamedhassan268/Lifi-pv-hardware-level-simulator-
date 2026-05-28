"""POST /api/config/validate — round-trip a config through SystemConfig validation.

Returns both the legacy `{valid, errors[]}` shape (for older clients) and a
structured `issues[]` list (ERROR + WARNING + INFO) from cosim.validation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.schemas import ConfigValidateResponse, ValidationIssueModel
from cosim.system_config import SystemConfig
from cosim.validation import IssueLevel, validate_config

router = APIRouter()


def _issues_to_models(issues) -> list[ValidationIssueModel]:
    return [
        ValidationIssueModel(
            level=i.level.value,
            field=i.field,
            message=i.message,
            suggestion=i.suggestion,
            rule_id=i.rule_id,
        )
        for i in issues
    ]


@router.post("/validate", response_model=ConfigValidateResponse)
def validate_config_endpoint(payload: dict[str, Any]) -> ConfigValidateResponse:
    # Try to construct. Construction itself raises ValueError on ERROR-level
    # issues, so we run the validator manually on a best-effort cfg if that
    # happens.
    try:
        cfg = SystemConfig._from_flat_dict(payload)
    except ValueError as e:
        # Construction failed — run rules on a permissive copy so we can still
        # return per-field issues. Build a default cfg, then patch over it
        # with whatever payload keys are valid; surface ERROR rules.
        try:
            cfg = SystemConfig()
            for k, v in payload.items():
                # Route by checking which sub-config owns the field.
                for sub_name in ('tx', 'channel', 'rx', 'noise', 'sim',
                                 'modulation_config', 'validation'):
                    sub = getattr(cfg, sub_name)
                    if hasattr(sub, k):
                        setattr(sub, k, v)
                        break
            issues = validate_config(cfg)
        except Exception:
            return ConfigValidateResponse(
                valid=False,
                errors=str(e).splitlines(),
                issues=[ValidationIssueModel(
                    level="error",
                    field="config",
                    message=str(e),
                    rule_id="config.construction_failed",
                )],
            )
        return ConfigValidateResponse(
            valid=False,
            errors=str(e).splitlines(),
            issues=_issues_to_models(issues),
        )

    issues = validate_config(cfg)
    errors = [i.message for i in issues if i.level == IssueLevel.ERROR]
    return ConfigValidateResponse(
        valid=not errors,
        errors=errors,
        issues=_issues_to_models(issues),
        normalized=cfg.to_dict(),
    )
