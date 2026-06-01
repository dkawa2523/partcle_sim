# COMSOL Extraction Checklist

Use this before writing solver configs.

## Model identity

- [ ] COMSOL version recorded
- [ ] component selected
- [ ] study selected
- [ ] dataset selected
- [ ] solution index selected
- [ ] coordinate system recorded
- [ ] unit scale recorded

## Geometry

- [ ] mesh or boundary primitives exported
- [ ] boundary IDs exported
- [ ] part IDs mapped
- [ ] axis boundary marked if RZ
- [ ] open/outlet boundaries marked

## Fields

- [ ] field variables listed
- [ ] units recorded
- [ ] component directions mapped
- [ ] support/valid mask checked
- [ ] gas and temperature fields exported if needed

## Particles

- [ ] release table exported
- [ ] particle IDs available
- [ ] release times available
- [ ] initial velocities available
- [ ] long trajectory table exported if trajectory comparison is required
- [ ] sampled vs full labels recorded

## Physics

- [ ] force inventory complete
- [ ] wall laws complete
- [ ] stochastic policy recorded
- [ ] particle material properties available

## Comparison readiness

- [ ] allowed comparison layers listed
- [ ] missing layers listed
- [ ] stop conditions resolved or documented
