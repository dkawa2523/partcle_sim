# Installed editable examples

Examples are shipped in the distribution and generated through the same public
case publisher as user inputs. No checkout-only case writer is maintained here.

```console
python -m chamber_particles init work/xy --example xy-ballistic
python -m chamber_particles init work/rz --example rz-charged
python -m chamber_particles init work/time --example xy-time-field
python -m chamber_particles init work/noise --example brownian-wall
python -m chamber_particles init work/electric --example reduced-electrostatic
```

Each directory contains canonical `case.yaml`/`case.h5`, an editable realized
`particles.csv`, and `recipe.yaml`. Change the CSV or explicit settings, then
publish a new case with `python -m chamber_particles create recipe.yaml -o NEW_CASE`.
Examples are small analytic demonstrations, not device-specific defaults or a
physical prediction certification. The reduced electric example also includes
its thermal bundle and builder configuration.
