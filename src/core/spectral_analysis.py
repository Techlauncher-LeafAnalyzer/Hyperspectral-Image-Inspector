"""Full-spectrum analysis Model: region spectra, transforms and similarity.

Complements :mod:`core.visualization_model`, whose indices each read only two
or three bands, with operations over every band of the cube:

* region statistics -- the mean/std spectrum of each classification class or
  of the whole region of interest, read in band chunks so peak memory stays at
  one chunk of the cube rather than a full float32 copy;
* spectral transforms -- first derivative, continuum removal and area
  normalisation, applied to any spectrum before display;
* Spectral Angle Mapper (SAM) -- the per-pixel angle to a reference spectrum,
  and the pairwise angle between class means as a separability measure.

No Qt types are used here. Methods are synchronous; reads on one
:class:`HSIData` must be serialized by the Controller like every other read.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import numpy as np
from matplotlib import colormaps

from .errors import HSIError, VisualizationError
from .hsi_data import HSIData


# Bands read per chunk by the full-cube scans; bounds peak memory to
# rows * columns * _BAND_CHUNK float32 values.
_BAND_CHUNK = 32
# Reversed so the most similar (smallest-angle) pixels render brightest.
SIMILARITY_COLORMAP = "viridis_r"


class SpectrumTransform(StrEnum):
    """Display transforms applied to a spectrum before plotting."""

    NONE = "Reflectance"
    FIRST_DERIVATIVE = "First derivative"
    CONTINUUM_REMOVED = "Continuum removed"
    NORMALIZED = "Area normalized"


@dataclass(frozen=True, slots=True)
class RegionSpectrum:
    """Mean and spread of every band over one set of pixels.

    ``label`` is a display name (a class name or "Image mean"); ``key`` is a
    stable identifier the Controller can use for colours and caching.
    """

    key: str
    label: str
    pixel_count: int
    wavelengths_nm: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Per-pixel spectral angle (radians) to one reference spectrum.

    ``angles`` is NaN outside the region of interest. Smaller is more similar.
    ``display_rgb`` maps the 2nd-98th percentile angles through
    :data:`SIMILARITY_COLORMAP`; ``display_limits`` records that range.
    """

    reference_label: str
    angles: np.ndarray
    value_range: tuple[float, float]
    display_rgb: np.ndarray
    display_limits: tuple[float, float]


class SpectralAnalysisService:
    """Stateless full-spectrum computations over an :class:`HSIData` cube."""

    # ------------------------------------------------------------------ #
    # Region statistics                                                  #
    # ------------------------------------------------------------------ #

    def class_spectra(
        self,
        data: HSIData,
        class_map: np.ndarray,
        class_labels: Mapping[int, str],
    ) -> tuple[RegionSpectrum, ...]:
        """Return the mean/std spectrum of each class in ``class_labels``.

        ``class_map`` is the ``(rows, columns)`` integer map from a
        classification result; pixels whose value is not a key of
        ``class_labels`` (e.g. ``-1`` outside a polygon crop) are ignored.
        Classes with no pixels are omitted.
        """

        class_map = np.asarray(class_map)
        if class_map.shape != (data.rows, data.columns):
            raise VisualizationError(
                f"Class map shape {class_map.shape} does not match the "
                f"{data.rows}x{data.columns} image."
            )
        flat = class_map.reshape(-1)
        groups = {
            int(class_id): np.flatnonzero(flat == class_id)
            for class_id in class_labels
        }
        groups = {class_id: indices for class_id, indices in groups.items() if indices.size}
        stats = self._grouped_statistics(data, groups)
        return tuple(
            RegionSpectrum(
                key=f"class:{class_id}",
                label=class_labels[class_id],
                pixel_count=int(groups[class_id].size),
                wavelengths_nm=self._wavelengths(data),
                **stats[class_id],
            )
            for class_id in groups
        )

    def masked_spectra(
        self,
        data: HSIData,
        regions: Mapping[str, tuple[str, np.ndarray]],
    ) -> tuple[RegionSpectrum, ...]:
        """Return the mean/std spectrum over each boolean pixel mask.

        ``regions`` maps a stable key (e.g. ``"class:3"`` or
        ``"annotation:7"``) to ``(label, mask)``, where ``mask`` is a
        ``(rows, columns)`` boolean array. Classification classes and drawn
        annotation regions are therefore analysed alike, in one chunked pass
        over the cube. Pixels outside a polygon-crop region of interest are
        excluded; regions left with no pixels are omitted.
        """

        roi = None if data.roi_mask is None else np.asarray(data.roi_mask, dtype=bool)
        groups: dict[str, np.ndarray] = {}
        labels: dict[str, str] = {}
        for key, (label, mask) in regions.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (data.rows, data.columns):
                raise VisualizationError(
                    f"Mask for {label!r} has shape {mask.shape}; the image is "
                    f"{data.rows}x{data.columns}."
                )
            if roi is not None:
                mask = mask & roi
            indices = np.flatnonzero(mask.reshape(-1))
            if indices.size:
                groups[key] = indices
                labels[key] = label
        stats = self._grouped_statistics(data, groups)
        return tuple(
            RegionSpectrum(
                key=key,
                label=labels[key],
                pixel_count=int(groups[key].size),
                wavelengths_nm=self._wavelengths(data),
                **stats[key],
            )
            for key in groups
        )

    def image_spectrum(self, data: HSIData, label: str = "Image mean") -> RegionSpectrum:
        """Return the mean/std spectrum of every pixel in the region of interest."""

        if data.roi_mask is None:
            indices = np.arange(data.rows * data.columns)
        else:
            indices = np.flatnonzero(np.asarray(data.roi_mask).reshape(-1))
        if indices.size == 0:
            raise VisualizationError("The region of interest contains no pixels.")
        stats = self._grouped_statistics(data, {0: indices})[0]
        return RegionSpectrum(
            key="image",
            label=label,
            pixel_count=int(indices.size),
            wavelengths_nm=self._wavelengths(data),
            **stats,
        )

    # ------------------------------------------------------------------ #
    # Similarity                                                         #
    # ------------------------------------------------------------------ #

    def spectral_angle_map(
        self,
        data: HSIData,
        reference: np.ndarray,
        reference_label: str = "reference",
    ) -> SimilarityResult:
        """Return each pixel's spectral angle to ``reference``.

        The angle is independent of overall brightness, so shaded and lit
        parts of the same material score alike.
        """

        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference.size != data.bands:
            raise VisualizationError(
                f"Reference has {reference.size} bands; the image has {data.bands}."
            )
        reference_norm = float(np.linalg.norm(reference))
        if not np.isfinite(reference_norm) or reference_norm == 0:
            raise VisualizationError("The reference spectrum is all zero.")
        dot = np.zeros((data.rows, data.columns), dtype=np.float64)
        squared = np.zeros_like(dot)
        for start, chunk in self._band_chunks(data):
            chunk = np.nan_to_num(chunk.astype(np.float64, copy=False))
            weights = reference[start:start + chunk.shape[2]]
            dot += chunk @ weights
            squared += np.einsum("ijk,ijk->ij", chunk, chunk)
        norms = np.sqrt(squared) * reference_norm
        cosine = np.divide(dot, norms, out=np.ones_like(dot), where=norms > 0)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32)
        # A pixel with no signal has no defined angle.
        angles[norms <= 0] = np.nan
        angles = data.masked(angles).fill()
        finite = angles[np.isfinite(angles)]
        if finite.size:
            value_range = (float(finite.min()), float(finite.max()))
            low, high = (float(v) for v in np.percentile(finite, (2.0, 98.0)))
        else:
            value_range = (0.0, 0.0)
            low, high = 0.0, 0.0
        if high <= low:
            high = low + np.finfo(np.float32).eps
        normalized = np.nan_to_num(np.clip((angles - low) / (high - low), 0, 1))
        display = np.round(
            colormaps[SIMILARITY_COLORMAP](normalized)[..., :3] * 255
        ).astype(np.uint8)
        return SimilarityResult(reference_label, angles, value_range, display, (low, high))

    @staticmethod
    def spectral_angle(first: np.ndarray, second: np.ndarray) -> float:
        """Return the angle in radians between two spectra (NaN if either is zero)."""

        a = np.nan_to_num(np.asarray(first, dtype=np.float64))
        b = np.nan_to_num(np.asarray(second, dtype=np.float64))
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return float("nan")
        return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))

    def separability(self, spectra: Sequence[RegionSpectrum]) -> np.ndarray:
        """Return the symmetric matrix of spectral angles between region means."""

        count = len(spectra)
        matrix = np.zeros((count, count), dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                angle = self.spectral_angle(spectra[i].mean, spectra[j].mean)
                matrix[i, j] = matrix[j, i] = angle
        return matrix

    # ------------------------------------------------------------------ #
    # Transforms                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def transform(
        wavelengths_nm: np.ndarray,
        values: np.ndarray,
        transform: SpectrumTransform | str,
    ) -> np.ndarray:
        """Apply a display ``transform`` to one spectrum.

        Wavelengths must be in ascending order, as every supported camera
        reports them.
        """

        transform = SpectrumTransform(transform)
        x = np.asarray(wavelengths_nm, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        if x.shape != y.shape:
            raise VisualizationError("Wavelengths and values must have the same length.")
        if transform is SpectrumTransform.NONE or y.size < 2:
            return y
        if transform is SpectrumTransform.FIRST_DERIVATIVE:
            return np.gradient(y, x)
        if transform is SpectrumTransform.NORMALIZED:
            area = float(np.trapezoid(np.abs(y), x))
            return y / area if area > 0 else np.zeros_like(y)
        return y / _upper_hull(x, y)

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_csv(
        path: str | Path,
        wavelengths_nm: np.ndarray,
        series: Mapping[str, np.ndarray],
    ) -> Path:
        """Write one wavelength column plus one column per named spectrum."""

        output = Path(path)
        names = list(series)
        columns = [np.asarray(series[name]).reshape(-1) for name in names]
        wavelengths = np.asarray(wavelengths_nm).reshape(-1)
        if any(column.size != wavelengths.size for column in columns):
            raise VisualizationError("Every spectrum must have one value per wavelength.")
        try:
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["wavelength_nm", *names])
                for index, wavelength in enumerate(wavelengths):
                    writer.writerow(
                        [f"{wavelength:g}", *(f"{column[index]:.6g}" for column in columns)]
                    )
        except OSError as exc:
            raise VisualizationError(f"Could not write {output}: {exc}") from exc
        return output

    # ------------------------------------------------------------------ #
    # Private                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wavelengths(data: HSIData) -> np.ndarray:
        wavelengths = data.wavelengths_nm.copy()
        wavelengths.setflags(write=False)
        return wavelengths

    @staticmethod
    def _band_chunks(data: HSIData):
        for start in range(0, data.bands, _BAND_CHUNK):
            bands = list(range(start, min(start + _BAND_CHUNK, data.bands)))
            try:
                yield start, data.read_bands(bands)
            except HSIError:
                raise
            except MemoryError as exc:
                raise VisualizationError(
                    "Not enough memory to read the cube; crop the image first."
                ) from exc

    def _grouped_statistics(
        self, data: HSIData, groups: Mapping[Hashable, np.ndarray]
    ) -> dict[Hashable, dict[str, np.ndarray]]:
        """Per-band mean/std/min/max over each group of flat pixel indices."""

        stats = {
            key: {name: np.empty(data.bands, dtype=np.float32)
                  for name in ("mean", "std", "minimum", "maximum")}
            for key in groups
        }
        for start, chunk in self._band_chunks(data):
            flat = chunk.reshape(-1, chunk.shape[2])
            stop = start + chunk.shape[2]
            for key, indices in groups.items():
                pixels = flat[indices]
                with np.errstate(all="ignore"):
                    stats[key]["mean"][start:stop] = np.nanmean(pixels, axis=0)
                    stats[key]["std"][start:stop] = np.nanstd(pixels, axis=0)
                    stats[key]["minimum"][start:stop] = np.nanmin(pixels, axis=0)
                    stats[key]["maximum"][start:stop] = np.nanmax(pixels, axis=0)
        for arrays in stats.values():
            for array in arrays.values():
                array.setflags(write=False)
        return stats


def _upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the upper convex hull of ``(x, y)`` interpolated at every ``x``.

    Used as the continuum for continuum removal; the result never falls below
    ``y``, so ``y / hull`` lies in ``(0, 1]`` for positive spectra.
    """

    hull: list[int] = []
    for index in range(x.size):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b when it lies on or below the chord from a to index.
            cross = (x[b] - x[a]) * (y[index] - y[a]) - (y[b] - y[a]) * (x[index] - x[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(index)
    continuum = np.interp(x, x[hull], y[hull])
    return np.where(np.abs(continuum) > np.finfo(np.float64).eps, continuum, 1.0)
