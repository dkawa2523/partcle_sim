"""Two separate full-population panels per case, from completed saved paths."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tools.vv.comsol.plot_matched_trajectories import (
    PlotBox,
    SvgCanvas,
    _browser_path,
    _draw_axes,
    _load_geometry,
    _plot_ranges,
    _project,
    read_matched_trajectories,
)

HERE = Path(__file__).resolve().parent


def bind(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def draw_paths(canvas, table, box, x_range, y_range) -> int:
    points_count = 0
    for x, y in zip(table.r_by_particle, table.z_by_particle, strict=True):
        finite = np.isfinite(x) & np.isfinite(y)
        points_count += int(finite.sum())
        cuts = np.flatnonzero(np.diff(finite.astype(np.int8))) + 1
        for segment in np.split(np.arange(x.size), cuts):
            if segment.size and finite[segment[0]]:
                points = _project(x[segment] * 1000, y[segment] * 1000, x_range, y_range, box)
                canvas.polyline(points, stroke="#2563EB", width=0.85, opacity=0.35)
                if segment.size == 1:
                    canvas.circle(*points[0], 1.2, fill="#2563EB")
    return points_count


def draw_geometry(canvas, geometry, box, x_range, y_range) -> None:
    for edge in geometry.line2:
        points = _project(
            geometry.nodes_m[edge, 0] * 1000,
            geometry.nodes_m[edge, 1] * 1000,
            x_range,
            y_range,
            box,
        )
        canvas.polyline(points, stroke="#334155", width=1.6, opacity=0.9)
    for edge in geometry.axis_line2:
        points = _project(
            geometry.nodes_m[edge, 0] * 1000,
            geometry.nodes_m[edge, 1] * 1000,
            x_range,
            y_range,
            box,
        )
        canvas.polyline(points, stroke="#94A3B8", width=1.2, dash="3 5")


def rasterize(path: Path) -> Path:
    output = path.with_suffix(".png")
    with tempfile.TemporaryDirectory(
        prefix="particle_side_panels_", ignore_cleanup_errors=True
    ) as profile:
        subprocess.run(
            [
                str(_browser_path(None)),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--disable-background-mode",
                "--disable-sync",
                f"--user-data-dir={profile}",
                f"--screenshot={output}",
                "--window-size=1600,700",
                path.as_uri(),
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    if not output.is_file():
        raise RuntimeError("Raster image not published")
    return output


def render_case(name: str) -> dict:
    source = HERE / "trajectory_review" / name
    data = read_matched_trajectories(source / "rk4_fine.csv", source / "native10.csv")
    geometry = _load_geometry(HERE / name / "prepared/case.h5", "axisymmetric_rz")
    if geometry is None or data.candidate.particle_ids.tolist() != list(range(1, 288)):
        raise ValueError("Required geometry or complete registered cohort missing")
    canvas = SvgCanvas(1600, 700, f"{name}: all trajectories; solver left, COMSOL right")
    canvas.text(55, 42, f"ケース{name[-1]} | 全粒子の軌道を横並び比較", css_class="title")
    canvas.text(
        55,
        70,
        "各287粒子 / 保存期間 0-30 ms / 同じ座標範囲・縮尺 / r-z断面・単位mm",
        css_class="subtitle",
    )
    left, right = PlotBox(100, 175, 665, 345), PlotBox(890, 175, 665, 345)
    x_range, y_range = _plot_ranges(data, geometry, left)
    counts = []
    for box, table, label, method in (
        (left, data.candidate, "本ソルバー: 全287粒子", "RK4 / 計算刻み 0.125 µs"),
        (right, data.reference, "COMSOL: 全287粒子", "原モデルの保存結果 / RK4設定 10 µs"),
    ):
        canvas.text(box.x + box.width / 2, 117, label, css_class="title", anchor="middle")
        canvas.text(box.x + box.width / 2, 146, method, css_class="subtitle", anchor="middle")
        _draw_axes(canvas, box, x_range, y_range, x_label="r (mm)", y_label="z (mm)")
        counts.append(draw_paths(canvas, table, box, x_range, y_range))
        draw_geometry(canvas, geometry, box, x_range, y_range)
    canvas.text(
        55,
        593,
        "青線: 全粒子の保存軌道 / 灰線: チャンバー境界 / 点線: 非物質の対称軸",
        css_class="subtitle",
    )
    canvas.text(
        55,
        624,
        f"保存された有限位置点: 本ソルバー {counts[0]:,}点 / COMSOL {counts[1]:,}点。欠測区間や消失後の位置は補いません。",
        css_class="subtitle",
    )
    canvas.text(
        55,
        655,
        "原COMSOLと本ソルバーは物理モデル・場表現が異なります。線は保存点を結んだ表示で、未保存の衝突経路は再構成していません。",
        css_class="small",
    )
    output = HERE / f"{name}_all_paths_side_by_side.svg"
    canvas.write(output)
    png = rasterize(output)
    return {
        "case": name,
        "particles_each_panel": 287,
        "time_range_s": [0, 0.03],
        "position_points": {"solver": counts[0], "COMSOL": counts[1]},
        "x_range_mm": x_range,
        "y_range_mm": y_range,
        "equal_aspect": True,
        "input": [bind(data.candidate.path), bind(data.reference.path)],
        "geometry": bind(geometry.path),
        "case_result_binding": bind(source / "figure_case_result_binding.json"),
        "svg": bind(output),
        "png": bind(png),
    }


def main() -> None:
    results = [render_case(name) for name in ("caseA", "caseP")]
    body = "".join(
        f'<section><h2>ケース{item["case"][-1]}</h2><a href="{item["case"]}_all_paths_side_by_side.svg"><img src="{item["case"]}_all_paths_side_by_side.png" alt="本ソルバー左、COMSOL右の全287軌道"></a></section>'
        for item in results
    )
    page = (
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本ソルバーとCOMSOLの全軌道比較</title><style>body{font-family:system-ui,"Yu Gothic",sans-serif;margin:24px;background:#f1f5f9;color:#183147}section{background:white;border:1px solid #d5dfe9;margin:20px 0;padding:16px;border-radius:6px}h1{font-size:25px}h2{margin:0 0 12px}img{width:100%;height:auto;display:block}a{color:#145d92}</style><h1>本ソルバーとCOMSOLの全軌道を横並び比較</h1><p>左: 本ソルバー、右: COMSOL。各パネルに全287粒子の0-30 msの保存軌道を表示しています。画像をクリックするとSVGを拡大できます。</p>'
        + body
        + "</html>"
    )
    (HERE / "all_paths_side_by_side.html").write_text(page, encoding="utf-8")
    (HERE / "all_paths_side_by_side_receipt.json").write_text(
        json.dumps(
            {
                "scope": "All saved paths in separate side-by-side panels; no new simulation or particle subsampling",
                "line_width_px": 0.85,
                "line_opacity": 0.35,
                "shared_path_color": "#2563EB",
                "cases": results,
                "producer": bind(Path(__file__)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": [item["case"] for item in results],
                "particles_each_panel": 287,
                "page": str(HERE / "all_paths_side_by_side.html"),
            }
        )
    )


if __name__ == "__main__":
    main()
