# 02. COMSOL Particle Tracing Reference Model

この文書は、COMSOL Particle Tracing Module と比較するときに、sim_rev3側で何を一致対象にし、何を近似として扱うかを整理する。

## 1. COMSOL比較で合わせるべき層

COMSOL比較では、trajectory endpointだけを合わせない。以下の層に分ける。

```text
1. release feature semantics
2. initial position / velocity / time
3. coordinate system and scale
4. field and force sampling
5. time integration
6. wall event detection
7. wall condition behavior
8. stochastic ensemble behavior
```

## 2. 時間積分

COMSOL Particle Tracing Moduleでは、Newtonian first-order formulationの既定time steppingとしてDormand–Prince 5 Runge–Kutta、second-order Newtonian formulationの既定としてGeneralized alpha implicit methodが使われる。明示法はstiff問題に不向きな場合がある。

sim_rev3は高速化のためにETDやdrag relaxationを使ってよい。ただし、COMSOL比較では以下を分離する。

```text
force model差
integrator差
event detection差
field sampling差
output cadence差
```

## 3. Release

COMSOLにはRelease、Release from Grid、Release from Data File、Inletなど複数のrelease系featureがある。Release from Data Fileでは初期位置、初期速度、補助変数をテキストファイルから与えられる。

sim_rev3では release を単なる初期particles CSVとして扱わず、少なくとも次の情報を保持する。

```text
release_time_s
raw_position
raw_velocity
release_feature_id
source_boundary_id
source_part_id
projection_distance_m
capture_tolerance_m
inward_offset_m
coordinate_system
random_seed / random_stream
```

COMSOL faithful modeでは、release座標をsolver側で勝手にsnapしない。ズレがあればimport/export診断として扱う。

Production surface-release modeでは、パーツ表面由来粒子としてboundary classification, projection, inward offsetを許す。

## 4. Wall condition

COMSOLのWall nodeには、Freeze、Bounce、Stick、Disappear、Pass through、Diffuse scattering、Isotropic scattering、Mixed diffuse and specular reflection、General reflection などがある。

sim_rev3側では、wall policy名をCOMSOL寄りに揃える。内部実装名が違っても、比較・manifest上は次のような物理名を使う。

```text
freeze
stick
disappear
pass_through
bounce_specular
diffuse_scattering
isotropic_scattering
mixed_diffuse_specular
general_reflection
thermal_reemission
```

壁処理はsegment endpointではなく、hit-time state `(x_hit, v_hit, t_hit)` で行う。

## 5. Axisymmetric RZ

COMSOL 2D axisymmetric component では Axial Symmetry node が自動的に扱われる。sim_rev3では `axisymmetric_rz` を `cartesian_xy` の別名にしてはいけない。

最低限必要:

```text
coordinate_system = axisymmetric_rz
r-axis / z-axis name
r=0 axis boundary semantics
surface weighting option = 2πr
source sampling measure awareness
```

v_theta まで最初から実装しなくてよいが、state設計を壊さないようにする。

## 6. Stochastic physics

Brownianなどの確率過程は、1粒子の逐点一致ではなくensemble分布で比較する。

deterministic comparisonでは:

```text
Brownian off
random source fixed
charge fixed or explicit
particle properties fixed
```

ensemble comparisonでは:

```text
seed policy
distribution parameters
source weighting
first-passage CDF
state fraction time series
```
