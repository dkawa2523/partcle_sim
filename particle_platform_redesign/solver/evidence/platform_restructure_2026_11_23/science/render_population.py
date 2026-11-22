"""Render the completed 40-replica assessment; no trajectory generation."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np


parser = argparse.ArgumentParser(description="Render the archived40-replica population summary")
parser.add_argument("--assessment", type=Path, required=True)
parser.add_argument("--output-directory", type=Path, required=True)
parser.add_argument("--svg-only", action="store_true")
parser.add_argument("--browser", type=Path)
arguments = parser.parse_args()
ROOT = arguments.output_directory.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = arguments.assessment.resolve()
SEEDS = list(range(21, 61))


def artifact(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def txt(x, y, value, size=15, anchor="start", color="#172033"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}">{html.escape(str(value))}</text>'


assert SOURCE.is_file(), "wait for the completed population assessment"
assessment = json.loads(SOURCE.read_text(encoding="utf-8"))
cases = assessment["cases"]
assert set(cases) == {"A", "P"}
for tag, case in cases.items():
    assert case["seeds"] == SEEDS
    assert len(case["restricted_mean_active_lifetime_s"]["candidate"]["values_by_seed"]) == 40
    assert len(case["boundaries"]["33"]["candidate"]["values_by_seed"]) == 40

life = {tag: np.array(c["restricted_mean_active_lifetime_s"]["candidate"]["values_by_seed"])*1e6 for tag,c in cases.items()}
native_life = {tag: c["restricted_mean_active_lifetime_s"]["native"]*1e6 for tag,c in cases.items()}
boundary = {tag: np.array(c["boundaries"]["33"]["candidate"]["values_by_seed"]) for tag,c in cases.items()}
native_boundary = {tag: c["boundaries"]["33"]["native"] for tag,c in cases.items()}


def edges_for(values, native, width):
    joined = np.concatenate([*values.values(), np.array(list(native.values()))])
    low = np.floor(joined.min()/width)*width-width
    high = np.ceil(joined.max()/width)*width+width
    return np.arange(low, high+width*.5, width)


life_edges = edges_for(life,native_life,5.)
boundary_edges = edges_for(boundary,native_boundary,5.)-.5
life_hist = {tag:np.histogram(v,life_edges)[0] for tag,v in life.items()}
boundary_hist = {tag:np.histogram(v,boundary_edges)[0] for tag,v in boundary.items()}
life_ymax = max(5.,float(max(v.max() for v in life_hist.values())))
boundary_ymax = max(5.,float(max(v.max() for v in boundary_hist.values())))

svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><rect width="1600" height="1000" fill="white"/>',
    txt(45,39,"Population comparison without random-path matching",27),
    txt(45,67,"Blue: 40 solver replicas, seeds21-60. Orange: one saved COMSOL realization. Each replica starts with the same 287 particles."),
    txt(45,91,"Active curves use common saved times. Shading is empirical replica min/max, not a confidence or simultaneous95% band.")]


def axes(left,top,width,height,xlim,ylim,xlabel,ylabel):
    xmin,xmax=xlim;ymin,ymax=ylim
    def xy(x,y):return left+(x-xmin)/(xmax-xmin)*width,top+height-(y-ymin)/(ymax-ymin)*height
    svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#94a3b8"/>')
    for v in np.linspace(xmin,xmax,4):
        x,y=xy(v,ymin);svg.append(txt(x,y+23,f"{v:.3g}",13,"middle"))
    for v in np.linspace(ymin,ymax,3):
        x,y=xy(xmin,v);svg.append(txt(x-8,y+4,f"{v:.3g}",13,"end"))
    svg.append(txt(left+width/2,top+height+48,xlabel,14,"middle"))
    svg.append(txt(left,top-13,ylabel,13))
    return xy


def polyline(points,color,width=2,opacity=1):
    return f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'


def histogram(tag,values,native,edges,hist,ymax,left,top,label,unit):
    width,height=400,230
    xy=axes(left,top,width,height,(float(edges[0]),float(edges[-1])),(0.,ymax),unit,"replicas (of40)")
    for x0,x1,count in zip(edges[:-1],edges[1:],hist,strict=True):
        px,py=xy(x0,float(count));rx,ry=xy(x1,0)
        svg.append(f'<rect x="{px+1:.2f}" y="{py:.2f}" width="{max(0.,rx-px-2):.2f}" height="{ry-py:.2f}" fill="#93b6f7" stroke="#2563eb" stroke-width=".7"/>')
    nx,ny=xy(native,0);_,nt=xy(native,ymax)
    svg.append(f'<path d="M{nx:.2f},{ny:.2f}L{nx:.2f},{nt:.2f}" stroke="#d97706" stroke-width="3"/>')
    svg.append(txt(left,top-43,label,18))
    svg.append(txt(left,top+height+72,f"Native {native:.3f}; solver mean {values.mean():.3f}",14))


for row,tag in enumerate(("A","P")):
    case=cases[tag];top=190+row*390
    svg.append(txt(45,top-70,f"Case {tag}",22))
    clock=np.array(case["clock_s"])*1e6
    keep=clock<=1500+16*np.finfo(float).eps*1500
    times=clock[keep]
    native=np.array(case["native_active_count_curve"])[keep]/287
    mean=np.array(case["candidate_mean_active_count_curve"])[keep]/287
    low=np.array(case["candidate_minimum_active_count_curve"])[keep]/287
    high=np.array(case["candidate_maximum_active_count_curve"])[keep]/287
    xy=axes(90,top,400,230,(0.,1500.),(0.,1.),"time [us] (0-1.5ms detail)","active / released287")
    svg.append(txt(90,top-43,"Active fraction",18))
    band=[xy(x,y) for x,y in zip(times,low,strict=True)]+[xy(x,y) for x,y in zip(times[::-1],high[::-1],strict=True)]
    svg.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in band)}" fill="#c4d8fc" opacity=".65"/>')
    svg.append(polyline([xy(x,y) for x,y in zip(times,mean,strict=True)],"#2563eb",2.5))
    svg.append(polyline([xy(x,y) for x,y in zip(times,native,strict=True)],"#d97706",2.5))
    svg.append(txt(90,top+302,"Lines connect saved observations for display.",13))
    histogram(tag,life[tag],native_life[tag],life_edges,life_hist[tag],life_ymax,620,top,"Restricted mean active lifetime","cohort mean [us], horizon5ms")
    histogram(tag,boundary[tag],native_boundary[tag],boundary_edges,boundary_hist[tag],boundary_ymax,1150,top,"Retained capture at boundary33","particles of287 at z=22mm surface")

svg += [txt(45,936,"Histogram bin widths: 5us and5 particles, for display only, not agreement gates. Histogram heights count whole-cohort replicas."),
        txt(45,961,"Native disappearance cause stays unknown; it is not relabeled escape. Numerical capture is not certified physical adhesion."),
        txt(45,984,"The weak-limit model, field-transfer and finite-step limitations remain. No individual stochastic trajectory is used as a matching gate."),'</svg>']

path=ROOT/"population_summary.svg"
path.write_text("\n".join(svg),encoding="utf-8")
if arguments.svg_only:
    rasterizer={"status":"not requested"}
else:
    sys.path.insert(0,str(Path.cwd()))
    from tools.vv.comsol.plot_matched_trajectories import _browser_path, _rasterize
    rasterizer=_rasterize([path],_browser_path(arguments.browser))
receipt={"scope":"saved completed40-replica population display; no simulation",
    "assessment":artifact(SOURCE),"renderer":artifact(Path(__file__)),"seeds":SEEDS,"particles_per_replica":287,
    "model":"refined1 fixed_rk4_white_limit, h1us/base2/max5",
    "horizon_s":.005,"curve_display_horizon_s":.0015,"band":"empirical candidate replica min/max, not CI or simultaneous95% band",
    "histogram_bins":{"lifetime_width_us":5.,"boundary_count_width_particles":5.,"role":"display only; no pass/fail threshold"},
    "meanings":{"lifetime":"restricted mean active lifetime per whole cohort, not individual-particle histogram",
                "boundary":"retained numerical capture count on verified native/canonical boundary33; missing native disappearance cause not reclassified"},
    "rasterizer":rasterizer,"figures":[artifact(path)]+([] if arguments.svg_only else [artifact(path.with_suffix('.png'))])}
(ROOT/"population_summary_receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"figure":str(path if arguments.svg_only else path.with_suffix('.png')),"assessment_sha256":receipt['assessment']['sha256']}))
