# opencv-rgb-precommit

`opencv-rgb-precommit` is a small pre-commit hook for Python code that uses
OpenCV. It protects the channel-order contract without banning BGR images:

The project targets Python 3.10 syntax and keeps its development checks on a
Python 3.10 environment.

- BGR is valid and may be passed directly to `cv2.imencode`.
- If OpenCV decodes an image as BGR and that BGR value has no other consumer,
  converting it to RGB is rejected. Use `IMREAD_COLOR_RGB` instead.
- A value known to be RGB may not be passed directly to `cv2.imencode`, which
  expects BGR input. Convert it with `COLOR_RGB2BGR` first.

## Examples

This is rejected because the decoded BGR value is only an intermediate:

```python
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

A missing-image guard does not make the intermediate BGR useful, so this is
also rejected:

```python
image = cv2.imread(path, cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("missing image")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Read RGB directly:

```python
image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
image = cv2.imdecode(encoded, cv2.IMREAD_COLOR_RGB)
```

BGR encoding remains valid:

```python
image = cv2.imread(path, cv2.IMREAD_COLOR)
ok, encoded = cv2.imencode(".webp", image)
```

When the application has RGB data, convert explicitly before encoding:

```python
bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
ok, encoded = cv2.imencode(".webp", bgr)
```

The checker intentionally uses conservative local AST/data-flow tracking. It
does not guess the channel order of values whose origin is unknown.

## Install

Add the hook to `.pre-commit-config.yaml` and pin a release tag:

```yaml
repos:
  - repo: https://github.com/ternaus/opencv-rgb-precommit
    rev: v0.1.0
    hooks:
      - id: check-opencv-rgb
```

Run it locally with:

```bash
pre-commit run --all-files
```

## Development

```bash
uv sync --python 3.10 --group dev
uv run --python 3.10 --locked --group dev pytest
uv run --python 3.10 --locked --group dev ruff check .
uv run --python 3.10 --locked --group dev ruff format --check .
pre-commit run --all-files
```
