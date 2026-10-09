# HyperView application branding

The icon retains the reference's isometric hyperspectral data cube and unfolded
bands, adds a leaf for plant imaging, and uses a restrained orbital line and
four-point star to express twentieth-century Space Age optimism. The production
palette is cream, warm orange, petrol blue, navy, and muted sage.

## Assets

- `src/ui/assets/hyperview.png`: master artwork, transparent exterior, Qt window/application icon.
- `src/ui/assets/hyperview.ico`: Windows executable icon, 16/24/32/48/64/128/256 px.
- `src/ui/assets/hyperview.icns`: macOS application-bundle icon, up to 1024 px.

All native formats derive from the same master. Re-export after changing the PNG:

```sh
python scripts/export_app_icons.py
```

The artwork was generated with the built-in imagegen tool using the supplied
`src/ui/assets/logo.jpg` as a visual reference, then refined with the prompt below.
Format conversion only resamples and encodes the approved PNG; it does not redraw it.

## Packaging

Run `python scripts/build_pyinstaller.py` on each target OS. The script creates
a fresh `build/venv` and installs the runtime, super-resolution, and build
requirements automatically; `--reuse-venv` reuses it for packaging iterations.
PyInstaller cannot cross-compile. Linux archives use `.tar.xz`; Windows/macOS
archives use `.zip`.
The build uses the ICO on Windows and ICNS on macOS. macOS archives contain
`HyperView.app`; Windows/Linux archives contain the `HyperView` directory.
The complete assets folder remains bundled for runtime resource lookup.

Linux ELF files do not carry file-manager icons. To register a desktop launcher,
save the following as `~/.local/share/applications/HyperView.desktop`, replacing
both example paths with the absolute location of the extracted package.
The icon is bundled under PyInstaller 6's `_internal/ui/assets` folder.
Quote the executable path when it contains spaces.

```ini
[Desktop Entry]
Type=Application
Name=HyperView
Comment=Hyperspectral image inspection for plant imaging
Exec=/absolute/path/HyperView/HyperView
Icon=/absolute/path/HyperView/_internal/ui/assets/hyperview.png
Terminal=false
Categories=Science;Graphics;
StartupWMClass=HyperView
```

Qt's desktop-file name is `HyperView`, matching this launcher. On Windows the
explicit application ID is `leaf.HyperView`, so the taskbar uses the application's
identity and icon.

## Generation prompt

```text
Use case: logo-brand
Asset type: final desktop application icon for HyperView, a leaf/plant hyperspectral imaging inspector.
Input image: reference only, preserve its recognizable isometric stacked spectral data cube and three spectral slices unfolding to the right. Reinterpret, do not copy exactly.
Primary request: redesign the reference mark in twentieth-century Space Age retrofuturist visual language inspired by sophisticated 1950s–1970s science-fiction magazine covers and space exploration posters. The main motif remains a hyperspectral cube, with a clear stylized leaf integrated on the top plane (one leaf, clean bold shape and central vein), and three separated rounded spectral panels projecting to the right, conveying observation and spectral imaging. Smooth streamlined architectural volumes, curved capsule-like edges. One restrained elliptical orbital line around the cube and a single small warm-orange four-point star, subordinate to the cube and leaf.
Composition: one finished square application icon, centered generous large silhouette filling about 80% of canvas, comfortable padding, absolutely no mockup or contact sheet. Rounded-square solid deep navy badge with cream edge detail, genuinely transparent exterior outside the badge. Strong readable silhouette at 32 and 64 pixels, broad flat color areas, few details.
Palette: cream white, warm orange, petrol blue, deep navy, beige; controlled 5-color screenprint-like shading. Crisp clean vector-like edges, very subtle vintage warmth, no heavy distress, gradients or noisy textures.
Text: no text or letters, no watermark.
Avoid: cyberpunk neon, steampunk gears, stock galaxy backgrounds, realistic cosmos, unrelated subjects, rockets, planets as main subject, typography, tiny embellishments, busy scenery. Preserve the reference's scientific cube identity while making leaf and spectral exploration unmistakable.
```

## Final refinement prompt

```text
Use case: precise-object-edit. Final production desktop icon for HyperView. Edit this generated logo only: keep the excellent leaf on the rounded spectral cube, the three unfolded spectral panels, the rounded navy badge, the orbital swoosh and four-point star in their exact layout.
Correct the palette to authentic mid-century retro-futurist printed poster colors: replace ALL bright electric cyan/light blue on the bottom cube layer and outermost spectral panel with muted WARM BURNT ORANGE (#D77B42); use muted PETROL BLUE (#22616A) for the middle layer/panel, DEEP NAVY (#112D42), CREAM (#F3E5C8) edge lines and leaf highlights, and muted sage/petrol on the leaf (no neon green). The little viewing aperture also warm orange. Reduce gradients to broad clean flat color regions.
Essential cleanup: remove EVERY fleck, white grain, rough cutout residue or stray pixel outside the rounded-square badge. The exterior must be perfectly empty transparent alpha, with a smooth pristine silhouette all around. No texture or distress anywhere, no shadow outside the icon. Keep overall badge shape and content. This is an app icon, not a physical sticker or distressed print. No text, no watermark.
```

## Transparency cleanup prompt

```text
Use case: background-extraction. Production cleanup of the attached HyperView app icon. Preserve EXACTLY the colored artwork INSIDE the cream rounded-square outline, including the cream outline itself. Change only the exterior outside this outline. There are unwanted opaque white/cream distressed flecks above, below and beside the badge; REMOVE ALL of them completely to transparent alpha. The OUTER edge of the cream rounded-square border must be one continuous mathematically smooth curve without any grain or protrusions. Flat, perfectly clean, vector-like outer silhouette. Keep the existing leaf, hyperspectral cube, orange/petrol palette, orbit and star unchanged. Genuine transparency outside. Do not reinterpret, redraw, add texture or add a shadow. Output one pristine app icon with 5 percent clear transparent padding around the badge.
```
