"""Exploratory fixed-cohort replica prediction, never a physics-fit or parity gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from chamber_particles import open_result

parser = argparse.ArgumentParser(description="Frozen A/P287-particle evidence reanalysis; no simulation")
parser.add_argument("--assessment-root", type=Path, required=True)
parser.add_argument("--native-evidence-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
ROOT = arguments.assessment_root.resolve()
PRIOR = arguments.native_evidence_root.resolve()
OUT = arguments.output.resolve()
EXPECTED_SEEDS = list(range(21, 61))
HORIZON = .005
HASHES = {}


def bind(path):
    path = Path(path).resolve()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = value
    return value


def load_json(path):
    bind(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_bound_artifact(value):
    assert bind(value["path"]) == value["sha256"], value["path"]


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"mean":float(a.mean()), "sample_sd":float(a.std(ddof=1)),
            "monte_carlo_se_of_candidate_mean":float(a.std(ddof=1)/np.sqrt(a.size)),
            "minimum":float(a.min()), "maximum":float(a.max()),
            "quantiles_p05_p25_p50_p75_p95":list(np.quantile(a,[.05,.25,.5,.75,.95])),
            "values_by_seed":list(a)}


def scalar_rank(native, values):
    a=np.asarray(values)
    n=a.size
    less=int(np.count_nonzero(a<native));greater=int(np.count_nonzero(a>native));equal=n-less-greater
    lower=(1+less+equal)/(n+1)
    upper=(1+greater+equal)/(n+1)
    return {"native":float(native),"candidate":describe(a),
            "candidate_mean_minus_native":float(a.mean()-native),
            "candidate_mean_relative_to_native_minus_one":float(a.mean()/native-1) if native!=0 else None,
            "candidate_less_equal_greater":[less,equal,greater],
            "native_rank_interval_including_self":[less+1,less+equal+1],
            "lower_tail_inclusive_rank_probability":lower,
            "upper_tail_inclusive_rank_probability":upper,
            "two_sided_conservative_rank_probability":min(1.,2*min(lower,upper)),
            "native_inside_candidate_minmax":bool(a.min()<=native<=a.max()),
            "candidate_minmax_predictive_coverage_under_exchangeable_continuous_null":(n-1)/(n+1),
            "rank_scope":"exploratory pointwise replica-law compatibility, not equivalence or a multiple-metric confirmatory test"}


def global_rank(native, candidates, *, distance):
    # Integer forms of each replica minus the mean of all OTHER replicas.
    # This is symmetric in all 41 replicas and avoids floating tie artifacts.
    all_counts=np.vstack((native,candidates)).astype(np.int64)
    n=all_counts.shape[0]
    absolute=np.abs(n*all_counts-all_counts.sum(axis=0))
    if distance=="maximum":
        raw=absolute.max(axis=1)
        scale=(n-1)*287
    else:
        raw=absolute.sum(axis=1)
        scale=2*(n-1)*287
    p=float(np.count_nonzero(raw>=raw[0])/n)
    center=candidates.mean(axis=0)
    native_to_candidate=(abs(native-center).max()/287 if distance=="maximum"
                         else abs(native-center).sum()/(2*287))
    return {"native_score":float(raw[0]/scale),
            "candidate_leave_one_out_scores":list(raw[1:]/scale),
            "integer_score_numerators":[int(x) for x in raw],
            "score_denominator":int(scale),
            "upper_tail_conservative_rank_probability":p,
            "native_to_candidate_mean_distance":float(native_to_candidate),
            "meaning":"symmetric leave-one-out distance across one whole curve/composition; native participates symmetrically in calibration, never coefficient fitting",
            "scope":"exploratory global functional replica-law comparison, not a purpose-margin equivalence test"}


def native_observations(label):
    file=PRIOR/"native"/f"case{label}_particles_long.csv"
    bind(file)
    with file.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    clock=np.asarray(sorted({float(r["time_s"]) for r in rows if float(r["time_s"])<=HORIZON}))
    bytime={float(t):[] for t in clock}
    first={}
    for r in rows:
        first.setdefault(int(r["particle_id"]),r)
        if float(r["time_s"]) in bytime:
            bytime[float(r["time_s"])].append(r)
    assert len(first)==287 and all(len(r)==287 for r in bytime.values())
    assert all(int(float(r["native_final_status"])) in (2,3,4)
               and float(r["release_time_s"])<=float(r["stop_time_s"])<HORIZON
               for r in first.values())
    active=np.asarray([sum(int(float(r["native_status"]))==1 for r in bytime[float(t)]) for t in clock],dtype=int)
    capture=np.asarray([sum(int(float(r["native_status"])) in (2,3) for r in bytime[float(t)]) for t in clock],dtype=int)
    durations=np.asarray([min(float(r["stop_time_s"]),HORIZON)-float(r["release_time_s"]) for r in first.values()])
    assert np.isfinite(durations).all() and np.all(durations>=0)
    description=load_json(PRIOR/f"case{label}/population_refined1_rk4_white_limit_s21.json")["reference"]
    wall=load_json(PRIOR/f"case{label}/terminal_boundary_refined1_rk4_white_limit_s21.json")
    dest={str(k):v["reference"]["count"] for k,v in wall["capture_by_boundary"].items()}
    return {"clock":clock,"active":active,"capture":capture,
            "restricted_mean_active_lifetime_s":float(durations.mean()),
            "capture_count":sum(dest.values()),"capture_count_by_boundary":dest,
            "unknown_disappearance_count":description["final_counts"]["disappeared_unknown"],
            "active_count_at5ms":int(active[-1]),"description":description}


def public_counts(row,clock):
    validate_bound_artifact(row["case"])
    validate_bound_artifact(row["result_manifest"])
    directory=Path(row["result_manifest"]["path"]).parent
    result=open_result(directory)
    assert result.manifest["data_content_hash"]==row["data_content_hash"]
    assert not any(result.manifest["failure_reason_counts"].values())
    frames={float(f.time_s):f for f in result.iter_frames() if float(f.time_s)<=HORIZON}
    assert all(float(t) in frames for t in clock)
    active=np.asarray([np.count_nonzero(frames[float(t)].lifecycle==1) for t in clock])
    capture=np.asarray([np.count_nonzero(frames[float(t)].lifecycle==2) for t in clock])
    for early in row["early_active"].values():
        ti=int(np.argmin(abs(clock-early["time_s"])))
        assert clock[ti]==early["time_s"]
        assert int(active[ti])==early["active_count"]
    return active,capture


# No interim statistical decisions: both completion records and all 40 results
# are required before creating this analysis output.
for label in ("A","P"):
    assert (ROOT/"ensemble"/f"case{label}/completion.json").is_file(),f"case{label} incomplete"
all_results={label:load_json(ROOT/"ensemble"/f"case{label}/results.json") for label in ("A","P")}
assert all(sorted(r["seed"] for r in rows)==EXPECTED_SEEDS for rows in all_results.values())
assert not OUT.exists(), "analysis output already exists"
report={"scope":"exploratory 40-candidate-replica prediction against one saved native realization per case",
        "primary":"0–5ms restricted mean active lifetime of the fixed 287-ID cohort",
        "selected_on_prior_results":True,
        "not_claimed":["confirmatory equivalence","qualified native solver parity","physical validation","pathwise random matching"],
        "replica_assumption":"independent candidate seeds approximate independent draws; under compatibility null native and candidate whole-cohort replicas are exchangeable",
        "dependence":"within-replica IDs/time need not be iid for these replica ranks; A/P with the same seed/native keys may be coupled and are analyzed separately, never pooled as80 independent replicas",
        "multiple_metrics":"primary and exploratory secondary functional/scalar ranks reported separately; probabilities are not a simultaneous all-metric PASS",
        "prediction_vs_mean":"40-candidate min/max is a 39/41≈95.12% prediction interval under continuous exchangeable null, not a confidence interval for the ensemble mean; ties make coverage conservative",
        "cases":{}}

for label, rows in all_results.items():
    rows=sorted(rows,key=lambda r:r["seed"])
    completion=load_json(ROOT/"ensemble"/f"case{label}/completion.json")
    plan=load_json(ROOT/"ensemble"/f"case{label}/plan.json")
    native=native_observations(label)
    assert len({r["data_content_hash"] for r in rows})==1
    assert all(not any(r["failure_reason_counts"].values()) for r in rows)
    assert all(r["counts"]["pending"]==0 for r in rows)
    public=[public_counts(r,native["clock"]) for r in rows]
    active=np.stack([x[0] for x in public]);capture=np.stack([x[1] for x in public])
    keys=sorted(set(native["capture_count_by_boundary"])|{str(k) for r in rows for k in r["capture_count_by_boundary"]},key=int)
    n_dest=[native["capture_count_by_boundary"].get(k,0) for k in keys]+[native["unknown_disappearance_count"],native["active_count_at5ms"]]
    c_dest=np.asarray([[r["capture_count_by_boundary"].get(k,0) for k in keys]+[r["counts"]["escaped"],r["counts"]["active"]] for r in rows])
    assert sum(n_dest)==287 and np.all(c_dest.sum(axis=1)==287)
    boundary={k:scalar_rank(native["capture_count_by_boundary"].get(k,0),[r["capture_count_by_boundary"].get(k,0) for r in rows]) for k in keys}
    for r in boundary.values():
        r["candidate_mean_minus_native_pp"]=r["candidate_mean_minus_native"]*100/287
    early={}
    for target in (10,20,30,50,100,200,500):
        old=native["description"]["series"]
        ti=int(np.argmin([abs(x["time_s"]-target*1e-6) for x in old]))
        n=old[ti]
        values=[min(r["early_active"].values(),key=lambda v:abs(v["time_s"]-target*1e-6)) for r in rows]
        assert all(v["time_s"]==n["time_s"] for v in values)
        early[str(target)]={"time_s":n["time_s"],"active_count":scalar_rank(n["status_counts"]["active"],[v["active_count"] for v in values])}
        for field,key in (("active_speed_m_s","speed_m_s_quantiles"),("active_charge_number","charge_number_quantiles")):
            early[str(target)][field]={}
            for i,q in enumerate(("p10","p50","p90")):
                finite=[v[key][i] for v in values if v[key] is not None and v[key][i] is not None]
                record={"native":n[field][q],"candidate_replicas_with_finite_quantile":len(finite)}
                if n[field][q] is not None and len(finite)==len(rows):
                    record["rank"]=scalar_rank(n[field][q],finite)
                else:
                    record["rank"]=None
                    record["reason"]="one or more participants has no active samples; no fabricated quantile or rank"
                early[str(target)][field][q]=record
    report["cases"][label]={"completion":completion,"plan_identity":plan["source"],
       "seeds":EXPECTED_SEEDS,"uncensored_before5ms":all(r["uncensored_before5ms"] for r in rows),
       "candidate_final_active_counts":[r["counts"]["active"] for r in rows],
       "restricted_mean_active_lifetime_s":scalar_rank(native["restricted_mean_active_lifetime_s"],[r["restricted_mean_active_lifetime_s"] for r in rows]),
       "capture_count":scalar_rank(native["capture_count"],[r["capture_count"] for r in rows]),
       "capture_terminal_median_s":scalar_rank(native["description"]["capture_terminal_time_s"]["p50"],[r["capture_terminal_quantiles_s"][1] for r in rows]),
       "boundaries":boundary,"boundary_composition_categories":[*keys,"nonretained_terminal_unknown_vs_explicit_escape","active_at5ms"],
       "destination_composition_meaning":"observed retained stopping surface + nonretained terminal + still active; native unknown disappearance and candidate explicit escape share only the coarse nonretained category, never an asserted physical cause",
       "destination_composition_global_tv":global_rank(np.asarray(n_dest),c_dest,distance="sum"),
       "active_curve_global_maximum":global_rank(native["active"],active,distance="maximum"),
       "capture_curve_global_maximum":global_rank(native["capture"],capture,distance="maximum"),
       "clock_s":list(native["clock"]),"native_active_count_curve":list(map(int,native["active"])),
       "candidate_mean_active_count_curve":list(active.mean(axis=0)),
       "candidate_minimum_active_count_curve":list(map(int,active.min(axis=0))),
       "candidate_maximum_active_count_curve":list(map(int,active.max(axis=0))),
       "native_capture_count_curve":list(map(int,native["capture"])),
       "candidate_mean_capture_count_curve":list(capture.mean(axis=0)),
       "early_active":early,
       "coverage":"all requested native-common frames present; every active candidate observation counted; terminal omissions do not create fabricated kinematics"}

bind(Path(__file__))
report["files_sha256"]=HASHES
OUT.mkdir(exist_ok=False)
(OUT/"assessment.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({case:{"lifetime_native_s":v["restricted_mean_active_lifetime_s"]["native"],
 "lifetime_mean_s":v["restricted_mean_active_lifetime_s"]["candidate"]["mean"],
 "lifetime_prediction_range_s":[v["restricted_mean_active_lifetime_s"]["candidate"]["minimum"],v["restricted_mean_active_lifetime_s"]["candidate"]["maximum"]],
 "lifetime_rank_probability":v["restricted_mean_active_lifetime_s"]["two_sided_conservative_rank_probability"],
 "capture_count_rank_probability":v["capture_count"]["two_sided_conservative_rank_probability"],
 "global_active_curve_rank_probability":v["active_curve_global_maximum"]["upper_tail_conservative_rank_probability"],
 "global_destination_rank_probability":v["destination_composition_global_tv"]["upper_tail_conservative_rank_probability"]}
 for case,v in report["cases"].items()},indent=2))
