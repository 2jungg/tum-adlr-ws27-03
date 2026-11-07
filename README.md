# tum-adlr-ws27-03

## Installation

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the venv and install everything (uv also fetches the pinned Python):

```bash
uv sync --extra cpu   # or --extra gpu (CUDA, needs NVIDIA driver >= 580)
```

## Usage

```bash
uv run python <script>.py   # run in the venv
uv add <package>            # add a dependency (--dev for dev tools)
```

Always pass `--extra cpu` or `--extra gpu` to `uv sync`, otherwise it removes torch.
