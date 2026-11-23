# Curved-wall line2 convergence

Status: **PASS**

The public solver API was run for 17 analytic rays against an
inscribed line2 approximation of a one-metre circle at [16, 32, 64, 128]
facets.  First-hit time, hit point, wall normal, and specular post-impact
velocity were compared independently with the smooth-circle analytic result.

Observed RMS orders are `{"rms_normal": 0.9981355555685293, "rms_point": 2.003135563270743, "rms_reflected_velocity": 0.9967374967318079, "rms_time": 2.003135563270619}`.  This validates
mesh convergence of the supported straight-facet representation; it does not
claim exact curved-wall geometry or COMSOL native curved-element equivalence.
