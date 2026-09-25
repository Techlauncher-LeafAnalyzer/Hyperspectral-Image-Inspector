#!/usr/bin/env bash
# Compile the inspector with Nuitka.
#
# Usage:
#   scripts/build_nuitka.sh              # standalone (fast iteration loop)
#   scripts/build_nuitka.sh onefile      # single-file distributable
#
# Build standalone first and validate the Super-Resolution feature against
# it (see the verification steps below) before spending the extra time on
# a onefile build: onefile implies standalone plus a full extra packaging
# pass on top, so it roughly doubles the wall-clock cost of every iteration.
#
# The --nofollow-import-to exclusions below exist because a single lazy
# `import torch` inside the optional Super-Resolution feature
# (src/core/super_resolution_model.py:93) makes Nuitka's whole-program
# standalone analysis follow torch's *entire* import graph, including
# subsystems this app never uses. Verified by reading
# src/core/super_resolution_model.py, src/core/sr/msdformer.py, and
# src/core/sr/common.py in full: the only torch/scipy surface actually
# touched is torch.nn, torch.nn.functional.normalize, torch.cuda.is_available,
# torch.load(weights_only=True), torch.inference_mode, basic tensor ops, and
# scipy.ndimage.zoom. Each exclusion group below is commented with why it's
# dead code for this app specifically, so it's obvious which line to remove
# if a future change (e.g. adding torch.compile) needs one of these back.
#
# IMPORTANT: --nofollow-import-to only affects statically-visible imports
# made by *our own code*; it does not know that torch's own __init__.py (or
# torch.nn's own source) may unconditionally import one of these submodules
# for its own bookkeeping, even when we never call into it. If that happens,
# `import torch` itself raises ImportError at runtime (caught by the
# `except ImportError` in _load_model() and re-raised as "SR requires
# PyTorch and SciPy..." - a prior version of this list learned this the hard
# way: excluding torch._dynamo/torch.export/sympy/torch.jit broke `import
# torch` entirely in the compiled binary, because torch 2.14's eager-mode
# shape/meta-tensor plumbing and torch.nn's @torch.jit.unused decorators
# reach into those at import time). Only the subsystems below are kept
# excluded because they are genuinely self-contained optional features with
# no plausible reach from core torch/torch.nn import:
#   - torch.testing: internal test-parametrization fixtures. This is the one
#     confirmed by a real crash - the original unfiltered build OOM-killed
#     gcc compiling torch.testing._internal.common_methods_invocations.
#   - torch.profiler, torch.distributed, torch.package, torch.ao,
#     torch.distributions: distinct, independently-loaded feature subsystems
#     that torch guards with is_available()/try-except internally, precisely
#     so builds without them keep working.
# Excluded previously but restored to avoid breaking plain `import torch`:
# torch._dynamo, torch._inductor, torch.fx, torch._higher_order_ops,
# torch._export, torch.export, torch._functorch, torchgen, sympy, mpmath,
# torch.jit. If compile time is still dominated by these after testing the
# trimmed list below, revisit them one at a time (not as a block) and
# verify SR still runs after each removal.

set -euo pipefail

MODE="${1:-standalone}"
if [[ "$MODE" != "standalone" && "$MODE" != "onefile" ]]; then
  echo "Usage: $0 [standalone|onefile]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TORCH_EXCLUDES=(
  # Internal test-parametrization fixtures - confirmed dead weight by an
  # actual crash: the original unfiltered build OOM-killed gcc compiling
  # torch.testing._internal.common_methods_invocations alone.
  --nofollow-import-to='torch.testing*'

  # SR inference is single-process; no torch.distributed.* call anywhere.
  # torch itself guards this with is_available()/try-except internally.
  --nofollow-import-to='torch.distributed*'

  # torch's own alternate archive/serialization format (PackageExporter),
  # unrelated to the plain torch.load() this app uses.
  --nofollow-import-to='torch.package*'

  # No ONNX export anywhere in this app.
  --nofollow-import-to='torch.onnx*'

  # MSDformer uses plain fp32 layers only: no quantized (torch.ao) layers,
  # no torch.distributions, no production profiler usage.
  --nofollow-import-to='torch.ao*'
  --nofollow-import-to='torch.distributions*'
  --nofollow-import-to='torch.profiler*'

  # NOT excluded (tried before, broke plain `import torch` at runtime):
  # torch._dynamo, torch._inductor, torch.fx, torch._higher_order_ops,
  # torch._export, torch.export, torch._functorch, torchgen, sympy, mpmath,
  # torch.jit - see the block comment above this array for why.
)

SCIPY_EXCLUDES=(
  # The only scipy call site is `from scipy.ndimage import zoom`
  # (super_resolution_model.py:94/193). scipy/ndimage/__init__.py does not
  # import these at module scope, so they're unreachable from that entry point.
  --nofollow-import-to='scipy.optimize*'
  --nofollow-import-to='scipy.sparse*'
  --nofollow-import-to='scipy.stats*'
  --nofollow-import-to='scipy.signal*'
  --nofollow-import-to='scipy.linalg*'
  --nofollow-import-to='scipy.io*'
  --nofollow-import-to='scipy.integrate*'
  --nofollow-import-to='scipy.spatial*'
)

python -m nuitka \
  --mode="$MODE" \
  --enable-plugin=pyqt6 \
  --include-package=spectral \
  --include-data-dir=model=model \
  --include-data-dir=src/ui/assets=src/ui/assets \
  "${TORCH_EXCLUDES[@]}" \
  "${SCIPY_EXCLUDES[@]}" \
  src/main.py
