"""Stable public Model API consumed by application Controllers."""

from .errors import (
    CancelledError,
    ClassificationError,
    HSIError,
    HSIFileError,
    HSIHeaderError,
    SuperResolutionError,
    VisualizationError,
    VisualizationExportError,
    WavelengthError,
)
from .classification_model import (
    ClassificationService,
    SupervisedClassificationRequest,
    SupervisedClassificationResult,
    SupervisedClassifierType,
    TrainingFilePair,
    TrainingPairResolver,
    UnsupervisedClassificationRequest,
    UnsupervisedClassificationResult,
    load_binary_training_mask,
)
from .hsi_data import Functionality, HSIData, ImageFormat
from .hsi_reader import HSIReader
from .super_resolution_model import (
    SuperResolutionRequest,
    SuperResolutionResult,
    SuperResolutionService,
)
from .visualization_model import (
    DisplayStretch,
    HypercubeData,
    HypercubeViewData,
    SpectrumResult,
    VisualizationMode,
    VisualizationRequest,
    VisualizationResult,
    VisualizationService,
)
from .visualization_export_model import (
    ImageExportFormat,
    VisualizationExportRequest,
    VisualizationExportResult,
    VisualizationExportService,
)

__all__ = [
    "CancelledError",
    "ClassificationError",
    "ClassificationService",
    "DisplayStretch",
    "Functionality",
    "HSIData",
    "HSIError",
    "HSIFileError",
    "HSIHeaderError",
    "HSIReader",
    "HypercubeData",
    "HypercubeViewData",
    "ImageFormat",
    "ImageExportFormat",
    "SpectrumResult",
    "SuperResolutionError",
    "SuperResolutionRequest",
    "SuperResolutionResult",
    "SuperResolutionService",
    "SupervisedClassificationRequest",
    "SupervisedClassificationResult",
    "SupervisedClassifierType",
    "TrainingFilePair",
    "TrainingPairResolver",
    "VisualizationError",
    "VisualizationExportError",
    "VisualizationExportRequest",
    "VisualizationExportResult",
    "VisualizationExportService",
    "VisualizationMode",
    "VisualizationRequest",
    "VisualizationResult",
    "VisualizationService",
    "UnsupervisedClassificationRequest",
    "UnsupervisedClassificationResult",
    "WavelengthError",
    "load_binary_training_mask",
]
