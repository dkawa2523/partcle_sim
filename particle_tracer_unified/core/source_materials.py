from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .datamodel import SourcePreprocessResult
from .source_model_application import apply_source_models
from .source_resolution import resolve_source_parameters


def diagnostics_to_dataframe(result: SourcePreprocessResult) -> pd.DataFrame:
    return pd.DataFrame(list(result.diagnostics_rows))


def material_source_summary(result: SourcePreprocessResult) -> pd.DataFrame:
    rows = diagnostics_to_dataframe(result)
    if rows.empty:
        return pd.DataFrame(columns=['resolved_material_id', 'law_name', 'particle_count', 'suppressed_count', 'mean_final_speed_mps'])
    grouped = rows.groupby(['resolved_material_id', 'law_name'], dropna=False)
    return grouped.agg(
        particle_count=('particle_id', 'count'),
        suppressed_count=('release_enabled', lambda s: int((1 - pd.Series(s).astype(int)).sum())),
        mean_final_speed_mps=('final_speed_mps', 'mean'),
    ).reset_index()


def event_source_summary(result: SourcePreprocessResult) -> pd.DataFrame:
    rows = diagnostics_to_dataframe(result)
    if rows.empty or 'matched_events' not in rows.columns:
        return pd.DataFrame(columns=['event_name', 'match_count'])
    counts: dict[str, int] = {}
    for item in rows['matched_events'].astype(str):
        for name in [part for part in item.split(';') if part]:
            counts[name] = counts.get(name, 0) + 1
    return pd.DataFrame({'event_name': list(counts.keys()), 'match_count': list(counts.values())})


def write_source_summary(result: SourcePreprocessResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(result.source_model_summary)
    summary_payload['event_summary'] = result.event_summary
    (output_dir / 'source_model_summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    diagnostics_to_dataframe(result).to_csv(output_dir / 'source_particle_diagnostics.csv', index=False)
    material_source_summary(result).to_csv(output_dir / 'material_source_summary.csv', index=False)
    event_source_summary(result).to_csv(output_dir / 'source_event_summary.csv', index=False)


__all__ = (
    'apply_source_models',
    'diagnostics_to_dataframe',
    'event_source_summary',
    'material_source_summary',
    'resolve_source_parameters',
    'write_source_summary',
)
