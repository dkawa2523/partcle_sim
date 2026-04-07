# v28 implementation status

Implemented and verified:
- self-contained package import/run
- step-aware source law catalog
- step-aware output policy
- prepared runtime builder
- process-step and recipe-manifest event binding
- material/source preprocessing
- 2D and 3D synthetic examples
- Numba free-flight kernels
- geometry-aware wall shear provider with scalar/vector/probe priority

Not yet restored in this distributable line:
- COMSOL mphtxt provider integration
- exact 3D BVH collision stack from earlier experimental branches
- advanced plasma charging / ion-drag branch from older experimental packages

Those are best reintroduced on top of this stabilized packaging line.
