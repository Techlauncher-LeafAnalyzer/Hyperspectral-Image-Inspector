"""CSV export of per-class classification summaries.

Produces one tidy table -- one row per class, one column per quantity -- so
the file loads directly into Excel, pandas, R, or similar without reshaping.
No Qt types are used here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .classification_layer_model import ClassIndexStatistics, ClassificationLayer
from .errors import ClassificationError

# Value of one visualization's class mean: a scalar index mean, or the mean
# (R, G, B) triple for the RGB visualization.
MeanValue = float | Sequence[float]

_RGB_CHANNELS = ("R", "G", "B")
_LEADING_COLUMNS = ("class_id", "class_name", "pixel_count", "image_fraction")


def export_class_summary_csv(
    path: str | Path,
    layers: Iterable[ClassificationLayer],
    means: Mapping[int, Mapping[str, MeanValue]],
    total_pixels: int,
    statistics: Mapping[int, Mapping[str, ClassIndexStatistics]] | None = None,
) -> Path:
    """Write each class's name, size, and visualization statistics to ``path``.

    Columns: ``class_id``, ``class_name``, ``pixel_count``,
    ``image_fraction`` (share of all ``total_pixels``), then per
    visualization ``<name>_mean`` (e.g. ``NDVI_mean``) -- or
    ``RGB_R_mean``/``RGB_G_mean``/``RGB_B_mean`` for RGB. Visualizations
    with ``statistics`` also get ``_median``, ``_std`` (population standard
    deviation), ``_min``, ``_max`` and ``_valid_pixels`` (pixels with a
    finite value, the sample size behind the other columns).

    Numbers are written unformatted at full precision; a class with no
    finite values for a visualization leaves those cells empty. The file is
    UTF-8 with a BOM so Excel reads non-ASCII class names correctly.

    Raises :class:`ClassificationError` if the file cannot be written.
    """

    layers = tuple(layers)
    statistics = statistics or {}

    # Visualizations in first-seen order, which follows the app's mode order.
    names: list[str] = []
    for source in (means, statistics):
        for per_class in source.values():
            for name in per_class:
                if name not in names:
                    names.append(name)

    columns: list[str] = []
    rows = []
    for layer in layers:
        row: dict[str, object] = {
            "class_id": layer.class_id,
            "class_name": layer.name,
            "pixel_count": layer.pixel_count,
            "image_fraction": layer.pixel_count / total_pixels if total_pixels else "",
        }
        for name in names:
            mean = means.get(layer.class_id, {}).get(name)
            stats = statistics.get(layer.class_id, {}).get(name)
            if mean is not None:
                if isinstance(mean, (int, float)):
                    row[f"{name}_mean"] = float(mean)
                else:
                    for channel, value in zip(_RGB_CHANNELS, mean):
                        row[f"{name}_{channel}_mean"] = float(value)
            if stats is not None:
                row[f"{name}_median"] = _cell(stats.median)
                row[f"{name}_std"] = _cell(stats.standard_deviation)
                row[f"{name}_min"] = _cell(stats.minimum)
                row[f"{name}_max"] = _cell(stats.maximum)
                row[f"{name}_valid_pixels"] = stats.finite_pixel_count
        rows.append(row)
        # Keep each visualization's columns together, in insertion order.
        columns.extend(column for column in row if column not in columns
                       and column not in _LEADING_COLUMNS)

    columns = _order_columns(columns, names)

    destination = Path(path)
    try:
        with destination.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[*_LEADING_COLUMNS, *columns], restval=""
            )
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise ClassificationError(f"Could not write {destination}: {exc}") from exc
    return destination


def _cell(value: float | None) -> float | str:
    return "" if value is None else value


_SUFFIX_ORDER = ("median", "std", "min", "max", "valid_pixels")


def _order_columns(columns: list[str], names: list[str]) -> list[str]:
    """Group columns by visualization: mean(s) first, then the other statistics."""

    def key(column: str) -> tuple[int, int, int]:
        for position, name in enumerate(names):
            if column.startswith(f"{name}_"):
                rest = column[len(name) + 1 :]
                if rest in _SUFFIX_ORDER:
                    return position, 1 + _SUFFIX_ORDER.index(rest), 0
                return position, 0, columns.index(column)
        return len(names), 0, columns.index(column)

    return sorted(columns, key=key)
