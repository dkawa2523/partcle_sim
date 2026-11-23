# External verification tools

[COMSOL tools](comsol/README.md) own native observation bindings, explicit
coefficient preparation, saved trajectory/population comparison and the small
native/analytic motion benchmarks. Normal input/run/result operations belong
to the installed `chamber-particles` package.

```console
uv run --locked python -m pytest tools/vv/comsol/tests -q
```

The suite uses temporary synthetic data and the tracked small native XY
reference. These tests require neither `model_dataset` nor a COMSOL license.
Fresh native benchmark generation requires COMSOL; comparison against saved
measurements and analytic references does not. Passing this suite is scoped
verification, not general native or physical-model certification.
