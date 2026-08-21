# ui_tests/resources/

Optional real hyperspectral captures for `ui_tests/ui/test_sample_images.py`.

These files are too large to commit, so `.gitignore` excludes everything in
this folder except this README. The rest of `ui_tests/` runs entirely against
a small synthetic cube generated in `conftest.py` and needs nothing here.

To exercise the app against a real capture, drop an ENVI header/data pair
anywhere under this folder, e.g.:

```
ui_tests/resources/
└── my-capture/
    ├── image.hdr
    └── image.bil        # or .bip/.bsq/.dat/.img/.raw
```

The header and data file must share the same filename stem (that's how
`HSIReader` pairs them). `test_sample_images.py` recursively discovers every
such pair under this folder and runs the load/visualize/crop/save checks
against each one. If this folder has no valid pairs, those tests are skipped
automatically — nothing needs to be configured.
