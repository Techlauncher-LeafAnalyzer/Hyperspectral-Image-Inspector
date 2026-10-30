"""Editor for the Visualization tab's user-defined Custom Index formula.

The user types a formula (or builds it from the palette buttons) and sees it
typeset like an equation editor would, together with the image's available
wavelength range and the actual band each ``R<nm>`` term will read. Only a
formula that parses and fits the current image can be applied.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class SavedFormula:
    """One named entry in the user's custom index library."""

    name: str
    formula: str


class CustomIndexStore:
    """Persist the user's named formulas and the applied one between sessions.

    The library is kept in insertion order. Names are matched ignoring case,
    so "my ndre" overwrites "My NDRE" rather than adding a near-duplicate.
    """

    KEY = "visualization/customIndexFormula"
    NAME_KEY = "visualization/customIndexName"
    LIBRARY_KEY = "visualization/customIndexLibrary"
    # Name given to a formula saved before the library existed.
    MIGRATED_NAME = "My custom index"

    def __init__(self, settings: QtCore.QSettings | None = None) -> None:
        self._settings = settings or QtCore.QSettings(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            "HyperView",
            "HyperView",
        )

    def load(self) -> str | None:
        """The applied formula, if any."""
        value = self._settings.value(self.KEY, "", type=str)
        return value.strip() or None

    def active_name(self) -> str | None:
        """The library name of the applied formula, if it was a saved one."""
        value = self._settings.value(self.NAME_KEY, "", type=str)
        return value.strip() or None

    def save(self, formula: str | None, name: str | None = None) -> None:
        """Record the applied formula and, when it is a saved one, its name."""
        for key, value in ((self.KEY, formula), (self.NAME_KEY, name)):
            if value:
                self._settings.setValue(key, value)
            else:
                self._settings.remove(key)
        self._settings.sync()

    def library(self) -> tuple[SavedFormula, ...]:
        """Every saved formula, oldest first."""
        if not self._settings.contains(self.LIBRARY_KEY):
            # First run with a library: keep a formula from an older version.
            formula = self.load()
            return (SavedFormula(self.MIGRATED_NAME, formula),) if formula else ()
        raw = self._settings.value(self.LIBRARY_KEY, "", type=str)
        try:
            entries = json.loads(raw)
            return tuple(
                SavedFormula(str(entry["name"]), str(entry["formula"]))
                for entry in entries
                if str(entry.get("name", "")).strip() and str(entry.get("formula", "")).strip()
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            return ()

    def find(self, name: str) -> SavedFormula | None:
        key = name.strip().casefold()
        return next((entry for entry in self.library() if entry.name.casefold() == key), None)

    def name_for(self, formula: str) -> str | None:
        """The name of a saved entry with exactly this formula, if any."""
        text = formula.strip()
        return next((entry.name for entry in self.library() if entry.formula == text), None)

    def save_entry(self, name: str, formula: str) -> SavedFormula:
        """Add a formula, or replace the one with the same name in place."""
        entry = SavedFormula(name.strip(), formula.strip())
        if not entry.name or not entry.formula:
            raise ValueError("A saved formula needs a name and a formula.")
        key = entry.name.casefold()
        entries = list(self.library())
        for position, existing in enumerate(entries):
            if existing.name.casefold() == key:
                entries[position] = entry
                break
        else:
            entries.append(entry)
        self._write_library(entries)
        return entry

    def delete_entry(self, name: str) -> None:
        key = name.strip().casefold()
        self._write_library([e for e in self.library() if e.name.casefold() != key])
        if (self.active_name() or "").casefold() == key:
            # The applied formula stays applied; it is just no longer named.
            self._settings.remove(self.NAME_KEY)
            self._settings.sync()

    def _write_library(self, entries: list[SavedFormula]) -> None:
        self._settings.setValue(
            self.LIBRARY_KEY,
            json.dumps([{"name": e.name, "formula": e.formula} for e in entries]),
        )
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
        *,
        store: CustomIndexStore | None = None,
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
        self._store = store
        self._formula: str | None = None
        self._formula_name: str | None = None

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

        if store is not None:
            layout.addWidget(self._build_library_row())

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
            "and invalid roots or logs show as 0. Save formulas you reuse with "
            "Save as…; the applied one is restored next time HyperView opens.",
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

        if store is not None:
            self.formulaEdit.textChanged.connect(self._sync_library_selection)

        self.formulaEdit.setText(formula or "")
        self._validate()
        self.formulaEdit.setFocus()

    @property
    def formula(self) -> str | None:
        """The applied formula text, or ``None`` before Apply."""
        return self._formula

    @property
    def formula_name(self) -> str | None:
        """The saved name of the applied formula, if it is in the library."""
        return self._formula_name

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

    def _build_library_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel("Saved formulas", row)
        self.savedCombo = QtWidgets.QComboBox(row)
        self.savedCombo.setObjectName("customIndexSavedCombo")
        self.savedCombo.setAccessibleName("Saved custom index formulas")
        self.savedCombo.setPlaceholderText("Unsaved formula")
        self.savedCombo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.savedCombo.setMinimumContentsLength(18)
        self.savedCombo.activated.connect(self._on_saved_formula_chosen)
        self.saveAsButton = QtWidgets.QPushButton("Save as…", row)
        self.saveAsButton.setObjectName("customIndexSaveAsButton")
        self.saveAsButton.setToolTip("Save this formula under a name to reuse it later")
        self.saveAsButton.setAutoDefault(False)
        self.saveAsButton.clicked.connect(self._on_save_as_clicked)
        self.deleteSavedButton = QtWidgets.QPushButton("Delete", row)
        self.deleteSavedButton.setObjectName("customIndexDeleteButton")
        self.deleteSavedButton.setToolTip("Remove the selected formula from your saved list")
        self.deleteSavedButton.setAutoDefault(False)
        self.deleteSavedButton.clicked.connect(self._on_delete_clicked)
        layout.addWidget(label)
        layout.addWidget(self.savedCombo, 1)
        layout.addWidget(self.saveAsButton)
        layout.addWidget(self.deleteSavedButton)
        # The selection syncs once the formula box is filled.
        self._reload_library(select="")
        return row

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

    def _reload_library(self, select: str | None = None) -> None:
        """Refill the saved-formula list, keeping the editor's text."""

        combo = self.savedCombo
        combo.blockSignals(True)
        combo.clear()
        for entry in self._store.library():
            combo.addItem(entry.name, entry.formula)
            combo.setItemData(combo.count() - 1, entry.formula, Qt.ItemDataRole.ToolTipRole)
        combo.blockSignals(False)
        if select is not None:
            combo.setCurrentIndex(combo.findText(select, Qt.MatchFlag.MatchFixedString))
        else:
            self._sync_library_selection()

    def _sync_library_selection(self, _text: str = "") -> None:
        """Show which saved formula the editor holds, or none if edited."""

        text = self.formulaEdit.text().strip()
        index = next(
            (i for i in range(self.savedCombo.count()) if self.savedCombo.itemData(i) == text),
            -1,
        )
        self.savedCombo.setCurrentIndex(index)
        self.deleteSavedButton.setEnabled(index >= 0)

    def _on_saved_formula_chosen(self, index: int) -> None:
        if index >= 0:
            self._set_formula_text(self.savedCombo.itemData(index))

    def _on_save_as_clicked(self) -> None:
        text = self.formulaEdit.text().strip()
        try:
            parse_custom_index(text)
        except CustomIndexError as exc:
            QtWidgets.QMessageBox.information(
                self, "Cannot save formula", f"Fix the formula first: {exc}"
            )
            return
        current = self.savedCombo.currentText() if self.savedCombo.currentIndex() >= 0 else ""
        name = self._ask_name(current or f"Custom index {len(self._store.library()) + 1}")
        if not name:
            return
        existing = self._store.find(name)
        if existing is not None and existing.formula != text and not self._confirm(
            "Replace saved formula?",
            f"“{existing.name}” already holds {existing.formula}. Replace it?",
        ):
            return
        saved = self._store.save_entry(name, text)
        self._reload_library(select=saved.name)
        self.deleteSavedButton.setEnabled(True)

    def _on_delete_clicked(self) -> None:
        index = self.savedCombo.currentIndex()
        if index < 0:
            return
        name = self.savedCombo.itemText(index)
        if not self._confirm("Delete saved formula?", f"Remove “{name}” from your saved formulas?"):
            return
        self._store.delete_entry(name)
        self._reload_library()

    def _ask_name(self, suggestion: str) -> str | None:
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Save formula", "Name for this formula:", text=suggestion
        )
        return name.strip() if accepted and name.strip() else None

    def _confirm(self, title: str, message: str) -> bool:
        answer = QtWidgets.QMessageBox.question(self, title, message)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

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
        self._formula_name = (
            self._store.name_for(self._formula) if self._store is not None else None
        )
        self.accept()
