"""Editor for the Visualization tab's user-defined Custom Index formula.

The user types a formula (or builds it from the palette buttons) and sees it
typeset like an equation editor would, together with the image's available
wavelength range and the actual band each ``R<nm>`` term will read. Only a
formula that parses and fits the current image can be applied.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.mathtext import math_to_image
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from core import CUSTOM_INDEX_FUNCTIONS, CustomIndexError, parse_custom_index

# Starting points offered in the Examples menu: (label, formula).
CUSTOM_INDEX_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("NDVI — normalized difference", "(R800 - R670) / (R800 + R670)"),
    ("Simple ratio", "R800 / R670"),
    ("Red-edge NDRE", "(R790 - R720) / (R790 + R720)"),
    ("Chlorophyll index (red edge)", "R800 / R720 - 1"),
    ("Anthocyanin reflectance index", "1/R550 - 1/R700"),
    ("Mixed example", "(1 + 0.25) / R800 + 8.0 * R700"),
)

# Palette buttons: (label, text inserted, cursor offset from the end, tooltip).
_OPERATOR_BUTTONS = (
    ("+", " + ", 0, "Add"),
    ("−", " - ", 0, "Subtract"),
    ("×", " * ", 0, "Multiply"),
    ("÷", " / ", 0, "Divide"),
    ("xʸ", "^", 0, "Power, e.g. R800^2 or R800^0.5"),
    ("( )", "()", 1, "Parentheses"),
)
_FUNCTION_BUTTONS = (
    ("√x", "sqrt()", 1, "sqrt"),
    ("x²", "sq()", 1, "sq"),
    ("|x|", "abs()", 1, "abs"),
    ("ln", "ln()", 1, "ln"),
    ("log₁₀", "log10()", 1, "log10"),
    ("eˣ", "exp()", 1, "exp"),
    ("min", "min(, )", 3, "min"),
    ("max", "max(, )", 3, "max"),
)

_PREVIEW_DPI = 130


class CustomIndexStore:
    """Persist the most recently applied formula between sessions."""

    KEY = "visualization/customIndexFormula"

    def __init__(self, settings: QtCore.QSettings | None = None) -> None:
        self._settings = settings or QtCore.QSettings(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            "HyperView",
            "HyperView",
        )

    def load(self) -> str | None:
        value = self._settings.value(self.KEY, "", type=str)
        return value.strip() or None

    def save(self, formula: str | None) -> None:
        if formula:
            self._settings.setValue(self.KEY, formula)
        else:
            self._settings.remove(self.KEY)
        self._settings.sync()


def render_formula_pixmap(latex: str, color: str = "#123e37") -> QtGui.QPixmap:
    """Typeset matplotlib mathtext as a transparent pixmap."""

    buffer = io.BytesIO()
    math_to_image(f"${latex}$", buffer, dpi=_PREVIEW_DPI, format="png", color=color)
    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap


class CustomIndexDialog(QtWidgets.QDialog):
    """Modal editor; read :attr:`formula` after ``exec()`` returns Accepted."""

    def __init__(
        self,
        wavelengths_nm: np.ndarray | None,
        formula: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("customIndexDialog")
        self.setWindowTitle("Custom Index")
        self.setModal(True)
        self.setMinimumWidth(640)
        self._wavelengths = (
            None
            if wavelengths_nm is None or np.size(wavelengths_nm) == 0
            else np.asarray(wavelengths_nm, dtype=np.float64)
        )
        self._formula: str | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Custom Index", self)
        heading.setObjectName("customIndexHeading")
        subtitle = QtWidgets.QLabel(
            "Write a formula using reflectance at any wavelength, written "
            "R&lt;nm&gt; — e.g. <b>R800</b>. Each wavelength is mapped to the "
            "nearest band this image has.",
            self,
        )
        subtitle.setObjectName("customIndexSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        self.rangeLabel = QtWidgets.QLabel(self._range_text(), self)
        self.rangeLabel.setObjectName("customIndexRange")
        self.rangeLabel.setWordWrap(True)
        layout.addWidget(self.rangeLabel)

        editor_row = QtWidgets.QHBoxLayout()
        self.formulaEdit = QtWidgets.QLineEdit(self)
        self.formulaEdit.setObjectName("customIndexFormulaEdit")
        self.formulaEdit.setAccessibleName("Custom index formula")
        self.formulaEdit.setPlaceholderText("(R800 - R670) / (R800 + R670)")
        self.formulaEdit.setClearButtonEnabled(True)
        examples = QtWidgets.QToolButton(self)
        examples.setObjectName("customIndexExamplesButton")
        examples.setText("Examples")
        examples.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        examples_menu = QtWidgets.QMenu(examples)
        for label, example in CUSTOM_INDEX_EXAMPLES:
            action = examples_menu.addAction(f"{label}    {example}")
            action.triggered.connect(
                lambda _checked=False, example=example: self._set_formula_text(example)
            )
        examples.setMenu(examples_menu)
        editor_row.addWidget(self.formulaEdit, 1)
        editor_row.addWidget(examples)
        layout.addLayout(editor_row)

        layout.addWidget(self._build_palette())
        layout.addWidget(self._build_band_inserter())

        self.previewLabel = QtWidgets.QLabel(self)
        self.previewLabel.setObjectName("customIndexPreview")
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previewLabel.setMinimumHeight(84)
        self.previewLabel.setAccessibleName("Formula preview")
        layout.addWidget(self.previewLabel)

        self.statusLabel = QtWidgets.QLabel(self)
        self.statusLabel.setObjectName("customIndexStatus")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.statusLabel)

        hint = QtWidgets.QLabel(
            "Operators + − × ÷ ^ and ( ); functions "
            + ", ".join(CUSTOM_INDEX_FUNCTIONS)
            + "; constants pi and e. Writing 2R800 multiplies. Division by zero "
            "and invalid roots or logs show as 0. The last applied formula is "
            "remembered next time HyperView opens.",
            self,
        )
        hint.setObjectName("customIndexHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.applyButton = self.buttonBox.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        self.applyButton.setText("Apply index")
        self.applyButton.setDefault(True)
        self.applyButton.setProperty("primaryButton", True)
        self.applyButton.clicked.connect(self._apply)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self._validate_timer = QtCore.QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(150)
        self._validate_timer.timeout.connect(self._validate)
        self.formulaEdit.textChanged.connect(lambda _text: self._validate_timer.start())
        self.formulaEdit.returnPressed.connect(self._apply)

        self.formulaEdit.setText(formula or "")
        self._validate()
        self.formulaEdit.setFocus()

    @property
    def formula(self) -> str | None:
        """The applied formula text, or ``None`` before Apply."""
        return self._formula

    # ------------------------------------------------------------------ #
    # Construction helpers                                               #
    # ------------------------------------------------------------------ #

    def _range_text(self) -> str:
        if self._wavelengths is None:
            return "Load an image to see its available bands."
        low, high = self._wavelengths.min(), self._wavelengths.max()
        count = self._wavelengths.size
        spacing = (
            float(np.median(np.abs(np.diff(np.sort(self._wavelengths)))))
            if count > 1 else 0.0
        )
        return (
            f"<b>Available bands:</b> {count} bands from {low:.1f} to "
            f"{high:.1f} nm (about {spacing:.2g} nm apart)."
        )

    def _build_palette(self) -> QtWidgets.QWidget:
        palette = QtWidgets.QWidget(self)
        palette.setObjectName("customIndexPalette")
        grid = QtWidgets.QGridLayout(palette)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        buttons = [
            (label, text, offset, tooltip) for label, text, offset, tooltip in _OPERATOR_BUTTONS
        ] + [
            (label, text, offset, CUSTOM_INDEX_FUNCTIONS[name])
            for label, text, offset, name in _FUNCTION_BUTTONS
        ]
        columns = 7
        for position, (label, text, offset, tooltip) in enumerate(buttons):
            button = QtWidgets.QToolButton(palette)
            button.setObjectName("customIndexPaletteButton")
            button.setText(label)
            button.setToolTip(f"{tooltip} — inserts {text.strip()}")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
            )
            button.clicked.connect(
                lambda _checked=False, text=text, offset=offset: self._insert(text, offset)
            )
            grid.addWidget(button, position // columns, position % columns)
        return palette

    def _build_band_inserter(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel("Band at", row)
        self.wavelengthSpin = QtWidgets.QDoubleSpinBox(row)
        self.wavelengthSpin.setObjectName("customIndexWavelengthSpin")
        self.wavelengthSpin.setAccessibleName("Wavelength to insert")
        self.wavelengthSpin.setDecimals(1)
        self.wavelengthSpin.setSuffix(" nm")
        self.wavelengthSpin.setSingleStep(1.0)
        if self._wavelengths is not None:
            self.wavelengthSpin.setRange(
                float(np.floor(self._wavelengths.min())),
                float(np.ceil(self._wavelengths.max())),
            )
            self.wavelengthSpin.setValue(
                float(np.clip(800.0, self.wavelengthSpin.minimum(), self.wavelengthSpin.maximum()))
            )
        else:
            self.wavelengthSpin.setRange(0.0, 5000.0)
            self.wavelengthSpin.setValue(800.0)
        self.nearestLabel = QtWidgets.QLabel(row)
        self.nearestLabel.setObjectName("customIndexNearestBand")
        insert = QtWidgets.QPushButton("Insert band", row)
        insert.setObjectName("customIndexInsertBandButton")
        insert.setAutoDefault(False)
        insert.clicked.connect(
            lambda: self._insert(f"R{self.wavelengthSpin.value():g}", 0)
        )
        self.wavelengthSpin.valueChanged.connect(self._update_nearest_label)
        layout.addWidget(label)
        layout.addWidget(self.wavelengthSpin)
        layout.addWidget(insert)
        layout.addWidget(self.nearestLabel, 1)
        self._update_nearest_label(self.wavelengthSpin.value())
        return row

    # ------------------------------------------------------------------ #
    # Behaviour                                                          #
    # ------------------------------------------------------------------ #

    def _update_nearest_label(self, value: float) -> None:
        if self._wavelengths is None:
            self.nearestLabel.setText("")
            return
        index = int(np.abs(self._wavelengths - value).argmin())
        self.nearestLabel.setText(
            f"→ nearest band {index} at {self._wavelengths[index]:.1f} nm"
        )

    def _set_formula_text(self, text: str) -> None:
        self.formulaEdit.setText(text)
        self.formulaEdit.setFocus()

    def _insert(self, text: str, cursor_back: int) -> None:
        edit = self.formulaEdit
        edit.insert(text)
        edit.setCursorPosition(edit.cursorPosition() - cursor_back)
        edit.setFocus()

    def _validate(self) -> bool:
        """Refresh the preview and status; return whether Apply is allowed."""

        self._validate_timer.stop()
        text = self.formulaEdit.text()
        try:
            formula = parse_custom_index(text)
        except CustomIndexError as exc:
            self.previewLabel.setText("Your formula will appear here")
            self._show_error(str(exc) if text.strip() else "", exc.position)
            return False
        try:
            self.previewLabel.setPixmap(render_formula_pixmap(formula.latex()))
        except Exception:  # Typesetting is cosmetic; never block a valid formula.
            self.previewLabel.setText(formula.text)
        if self._wavelengths is None:
            self._show_ok("Formula is valid. Load an image to map its bands.")
            return True
        try:
            resolved = formula.resolve(self._wavelengths)
        except CustomIndexError as exc:
            self._show_error(str(exc), None)
            return False
        mapping = "<br>".join(
            f"<b>{band.token}</b> → band {band.band_index} at "
            f"{band.wavelength_nm:.1f} nm"
            + (f" ({band.offset_nm:+.1f} nm)" if abs(band.offset_nm) >= 0.05 else "")
            for band in resolved
        )
        self._show_ok(mapping)
        return True

    def _show_error(self, message: str, position: int | None) -> None:
        self.statusLabel.setProperty("state", "error" if message else "")
        text = message.replace("<", "&lt;")
        formula = self.formulaEdit.text()
        if message and position is not None and position < len(formula):
            # Point at the offending character beneath the message.
            text += (
                f"<br><code>{formula[:position].replace('<', '&lt;')}"
                f"<span style='background:#f6c9c4;'>"
                f"{formula[position].replace('<', '&lt;')}</span>"
                f"{formula[position + 1:].replace('<', '&lt;')}</code>"
            )
        self.statusLabel.setText(text)
        self._repolish(self.statusLabel)
        self.applyButton.setEnabled(False)

    def _show_ok(self, message: str) -> None:
        self.statusLabel.setProperty("state", "ok")
        self.statusLabel.setText(message)
        self._repolish(self.statusLabel)
        self.applyButton.setEnabled(True)

    @staticmethod
    def _repolish(widget: QtWidgets.QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply(self) -> None:
        if not self._validate():
            return
        self._formula = self.formulaEdit.text().strip()
        self.accept()
