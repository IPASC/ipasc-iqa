# ipasc_iqa

**A benchmarking framework for Image Quality Assessment (IQA) of photoacoustic image reconstructions.**

`ipasc_iqa` evaluates IQA measures on standardized photoacoustic datasets, correlates them against
expert quality ratings, and ships a command-line interface plus a browser-based GUI. It is
reproducibility-first (every result records dataset, file hashes, and measure versions) and
ecosystem-friendly (third-party measures and datasets plug in via Python entry points).

See [`docs/IMPLEMENTATION_ROADMAP.md`](docs/IMPLEMENTATION_ROADMAP.md) for the full design.

---

## Installation

Requires **Python ≥ 3.12**.

```bash
pip install ipasc_iqa            # core (CLI + classical measures + PyTorch-based measures)
pip install "ipasc_iqa[gui]"     # + the web GUI 
```

NumPy 2.x is currently not part of the tested environment for deep metrics.
Use numpy>=1.26,<2 unless you are testing newer torch builds explicitly.
You can verify that the environment is correctly configured with:
```bash
ipasc_iqa doctor              # check core dependencies
ipasc_iqa doctor --gui        # also check GUI dependencies
ipasc_iqa doctor --deep       # also check deep-metric dependencies
ipasc_iqa doctor --gui --deep # check the full installation
```

Working from a checkout (recommended for development):

```bash
git clone https://github.com/IPASC/ipasc-iqa.git
cd ipasc-iqa
python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev,gui]"
```

---

## Quick start (CLI)

```bash
ipasc_iqa list measures
ipasc_iqa manifest generate path/to/dataset          # build a manifest.json
ipasc_iqa verify path/to/dataset                     # check file integrity
ipasc_iqa run \
    --dataset path/to/dataset \
    --measure psnr --measure ssim \
    --reference-preference ground_truth,pseudo_reference \
    --output results/run.json
ipasc_iqa compare results/run_a.json results/run_b.json
```

---

## Web application

### Launch locally

The GUI is a thin [FastAPI](https://fastapi.tiangolo.com/) + HTMX layer over the same Python API
the CLI uses. Install the `gui` extra first (`pip install -e ".[gui]"`).

```bash
ipasc_iqa-gui --workspace ./workspace --data-dir ./data --port 8000
```

Then open <http://127.0.0.1:8000>.

### Create the first administrator

To create the first administrator account, use:

```bash
ipasc_iqa-gui create-admin --email admin@example.org --name "admin_name" --password "admin_password" --workspace ./workspace --data-dir ./data
```

If the user has already registered through the web interface, the same command can be used without
`--name` and `--password` to promote the existing account.

| Flag | Meaning | Default |
|---|---|---|
| `--workspace` | Where run results are written (a `results/` folder is created here). | current directory |
| `--data-dir`  | Managed directory where datasets are downloaded and converted. | platform data dir |
| `--host`      | Bind address. | `127.0.0.1` |
| `--port`      | Bind port. | `8000` |
| `--reload`    | Auto-reload on code changes (development). | off |

The same two paths can be set via the `IPASC_IQA_WORKSPACE` and `IPASC_IQA_DATA_DIR` environment
variables.

### Main workflows

- **Run benchmark** — select datasets and compatible IQA measures, then compute image-level metric values.
- **Detailed analysis** — inspect metric results, compare reconstruction variants, and analyse agreement with human quality assessments.
- **Standings** — compare IQA measures through dataset- and task-specific rankings.
- **Contribute annotations** — contribute pairwise comparisons or absolute quality ratings.
- **Download** — export metric results and human annotations as JSON.

Because the GUI imports `ipasc_iqa` as a library, deploying it as a web service later is a matter of
running it behind an ASGI server (e.g. `uvicorn`/`gunicorn`) and a reverse proxy.

### Deployment notes

**Run a single worker.** `ipasc_iqa-gui` does this by default. Ranking sessions in progress, the
single-flight guard around result recomputation, and the rendered-image cache all live in process
memory, so a second worker would hand raters "session expired" at random and duplicate the heavy
recomputes. A single worker serves concurrent annotators fine — requests are handled on a thread
pool and the store (SQLite via `diskcache`) is safe for concurrent readers and writers. If the
deployment ever does need multiple workers, those three pieces have to move into the shared store
first.

Tuning knobs (environment variables, all optional):

| Variable | Meaning | Default |
|---|---|---|
| `IPASC_IQA_WORKSPACE` | Where results, annotations and accounts live. | current directory |
| `IPASC_IQA_DATA_DIR` | Managed dataset directory. | platform data dir |
| `IPASC_IQA_COOKIE_SECURE` | Set to `1` behind HTTPS so the session cookie is TLS-only. | `0` |
| `IPASC_IQA_RESULT_STALE_TTL` | Seconds a computed analysis/standings view keeps being served after new annotations arrive. Every contribution invalidates the results, so recomputing eagerly would mean re-running the correlations continuously during a study; instead views are served with a "Last updated" stamp and refreshed at most this often. `0` recomputes on every change. | `3600` |
| `IPASC_IQA_IMAGE_CACHE_BYTES` | Memory budget for rendered annotation PNGs. Raise it for large studies (each image is rendered once, then served from memory and revalidated by the browser via `ETag`). | `268435456` (256 MB) |

---

## Developing in PyCharm

1. **Open the project** — *File → Open* and select the cloned `ipasc-iqa` folder.

2. **Create the interpreter** — *Settings → Project: ipasc-iqa → Python Interpreter →
   Add Interpreter → Add Local Interpreter*. Either:
   - select the existing `.venv\Scripts\python.exe` (if you created the venv above), or
   - create a new *Virtualenv* with a base Python ≥ 3.12.

3. **Install dependencies** — open the PyCharm *Terminal* (it activates the venv automatically) and
   run:
   ```bash
   pip install -e ".[dev,gui]"
   ```

4. **Run the tests** — set pytest as the test runner under *Settings → Tools → Python Integrated
   Tools → Testing → Default test runner: pytest*. Then right-click the `tests/` folder and choose
   *Run 'pytest in tests'*. From the terminal it is simply:
   ```bash
   pytest
   ```

5. **Launch the web app** — *Run → Edit Configurations → + → Python*:
   - **Module name** (not script): `ipasc_iqa.gui.server`
   - **Parameters**: `--workspace . --data-dir .data --port 8000`
   - **Python interpreter**: the project venv
   - **Working directory**: the project root

   Run it, then open <http://127.0.0.1:8000>. Add `--reload` to the parameters for auto-reload while
   editing templates/CSS.

---

## Contributing

Guides for the two extension points:

- [**Write a custom measure**](docs/guides/writing-a-measure.md) — add a new IQA metric and open a PR.
- [**Add a dataset**](docs/guides/adding-a-dataset.md) — write a dataset adapter and open a PR.

Before opening a pull request, please make sure `ruff check .` and `pytest` both pass.

---

## Contributors

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

- **Mervenur Akkus** — Erasmus MC, Netherlands
- **Bruna Cotrufo** — Politecnico di Torino, Italy
- **Janek Gröhl** — RWTH Aachen, Germany

---

## License

MIT — see [`LICENSE`](LICENSE).
