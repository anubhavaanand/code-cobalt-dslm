"""Build the Kaggle notebook for Cobalt Phase 0.

Run with the system python (stdlib only):
    python3 scripts/make_kaggle_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


md("""# Cobalt — Phase 0: prove the pipeline

**Before running:** Settings → Accelerator → **GPU T4 ×2**. Settings → Internet → **On**.

This is not the real model. It exists to prove every stage connects. Expected
runtime: **~1 minute** for `smoke`, **~2 h** per run for `kaggle`, **~30 h** for
`base` (which needs checkpoint/resume across sessions).

**Change `REPO_URL` in the next cell** to point at your fork.
""")

code('''REPO_URL = "https://github.com/YOUR_USER/dslm"   # <-- CHANGE THIS
WORK_DIR = "/kaggle/working/dslm"
PRESET   = "smoke"      # "smoke" = quick check, "kaggle" = real ablation run
MAX_STEPS = 600         # smoke preset default; kaggle preset wants 9000, base 30000
''')

md("""## 1. Get the code

Cloned fresh each session — Kaggle notebooks do not persist between runs, which is
exactly why checkpoints get uploaded as a dataset at the end.
""")

code('''import os, subprocess

if os.path.exists(WORK_DIR):
    subprocess.run(["rm", "-rf", WORK_DIR], check=True)

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK_DIR], check=True)
os.chdir(WORK_DIR)
print("cloned to", WORK_DIR)
''')

md("## 2. Install dependencies")

code('''%cd {WORK_DIR}
!pip install -q -r requirements.txt
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
''')

md("""## 3. Prepare data

Downloads TinyStories, trains a BPE tokenizer, and writes uint16 token shards.
~2 minutes for the default 20k stories.
""")

code('''%cd {WORK_DIR}
!python scripts/prepare_data.py --preset {PRESET}

# The tokenizer may end up smaller than the preset's vocab if the corpus is small.
# Read the actual value so the model matches it exactly.
from dslm.tokenizer import Tokenizer
VOCAB = Tokenizer.from_file("data/tokenizer.json").vocab_size
print("tokenizer vocab:", VOCAB)
''')

md("""## 4. Train

`--resume latest` picks up from the most recent checkpoint if one exists, so a
9-hour session timeout does not cost you the run. Delete `checkpoints/` to start
over.
""")

code('''%cd {WORK_DIR}
import os
latest = "checkpoints/latest.pt"
resume = "--resume latest" if os.path.exists(latest) else ""
print("resume flag:", resume or "(fresh start)")

!python -m dslm.train --preset {PRESET} --vocab-size {VOCAB} --total-steps {MAX_STEPS} {resume}
''')

md("## 5. Sample")

code("""%cd {WORK_DIR}
!python -m dslm.sample --ckpt checkpoints/latest.pt --tokenizer data/tokenizer.json --prompt "Once upon a time" --max-new-tokens 60
""")

md("""## 6. Keep the output

Kaggle wipes `/kaggle/working` when the session ends. Download the checkpoints from
the output panel, or save them as a dataset so the next session can `--resume`.
""")

code('''import os
for f in sorted(os.listdir("checkpoints")):
    p = os.path.join("checkpoints", f)
    print(f"{os.path.getsize(p)/1e6:8.1f} MB  {p}")
''')

md("""---

## Phase 0 exit condition

- [ ] bytes/token between 3.5 and 4.5
- [ ] training loss decreases monotonically
- [ ] val loss finite and below the initial loss
- [ ] generated text is grammatical English, even if nonsensical
- [ ] you can explain `block_size`, `grad_accum`, `warmup_steps`, and why
      `ignore_index=-1` is in the loss

That last one is the real gate. If you cannot explain it, do not start Phase 2 —
you will not be able to debug a bad run.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/Cobalt_Phase0.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")