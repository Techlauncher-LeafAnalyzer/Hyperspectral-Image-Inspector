"""Domain failures that Controllers can translate into user-facing messages.

Catch :class:`HSIError` at the worker/Controller boundary. Subclasses support
specific titles and recovery actions without parsing exception message text.
Unexpected exceptions should still be logged as programming/environment faults.
"""


class HSIError(RuntimeError):
    """Base exception for anticipated, user-presentable Model failures."""


class HSIFileError(HSIError):
    """The selected data/header pair is missing or inconsistent."""


class HSIHeaderError(HSIError):
    """The hyperspectral header is malformed or unsupported."""


class WavelengthError(HSIError):
    """Required wavelength metadata or coverage is unavailable."""


class VisualizationError(HSIError):
    """A visualization request is invalid or could not be computed."""


class VisualizationExportError(VisualizationError):
    """A displayed visualization could not be validated or saved."""


class ClassificationError(HSIError):
    """A classification request is invalid or could not be computed."""


class CancelledError(HSIError):
    """A Controller-requested cancellation stopped Model processing.

    Treat this as a neutral outcome: clear busy state, retain the previous
    visualization, and do not show an error dialog.
    """


class SuperResolutionError(HSIError):
    """The SR input, checkpoint, dependencies, or inference are invalid."""
