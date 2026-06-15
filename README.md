# TM_Project_Masters

## Authors

- Beatris Daicu - 20221854
- Diogo Carvalho - 20221935
- Ricardo Pereira - 20250343
- Yehor Malakhov - 20221691

## Installation

If you're using pip, create a virtual environment with Python 3.14 before
installing the packages to avoid conflicts with other versions of packages. If
you're using [uv](https://docs.astral.sh/uv/), this is automatically handled for
you.

The code has not been tested on other Python versions, the versions of the
packages in the `requirements*.txt`/`pyproject.toml` files might only be
compatible with Python 3.14. Running the code with other package versions or
Python versions could lead to unexpected errors.

The model file required to run the final notebook can be found on
<https://liveeduisegiunl-my.sharepoint.com/:u:/g/personal/20221935_novaims_unl_pt/IQAjK9HEkdKgSboICF2PRMpCAct_zUgjNpDo80b_DnWvOGs?e=onekge>.
This file must be placed inside
`src/results/transformer_outputs_bert-base-multilingual-cased`.

### NVIDIA GPUs

Choose depending on your installed CUDA version.

#### Using pip

```bash
pip install -r "requirements-nvidia-cu12.txt"
```

```bash
pip install -r "requirements-nvidia-cu13.txt"
```

#### Using uv

```bash
uv sync --extra nvidia-cu12
```

```bash
uv sync --extra nvidia-cu13
```

### AMD GPUs

#### Windows

Choose depending on your AMD GPU, if it's part of the Radeon RX 7000 series,
choose gfx110x, if it's part of the Radeon RX 9000 series, choose gfx120x.

##### Using pip

```bash
pip install -r "requirements-amd-windows-gfx110x.txt"
```

```bash
pip install -r "requirements-amd-windows-gfx120x.txt"
```

##### Using uv

```bash
uv sync --extra amd-windows-gfx110X
```

```bash
uv sync --extra amd-windows-gfx120X
```

#### Linux

##### Using pip

```bash
pip install -r "requirements-amd-linux.txt"
```

##### Using uv

```bash
uv sync --extra amd-linux
```

### CPU or MPS

#### Using pip

```bash
pip install -r "requirements-cpu-mps.txt"
```

#### Using uv

```bash
uv sync --extra cpu-mps
```
