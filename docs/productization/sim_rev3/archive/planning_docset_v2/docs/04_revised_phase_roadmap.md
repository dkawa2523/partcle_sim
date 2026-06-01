# 04. Revised Phase Roadmap

## v2で修正した点

v1ではCompare / V&Vが後半に置かれていた。これは元の課題に対して弱い。元の課題は「比較評価が不十分なままCodexが局所修正を重ねること」なので、比較基盤は最初に必要である。

## フェーズ一覧

### Phase 0: Repo audit / cleanup inventory

コード変更なし。古い探索枝、temporary diagnostics、over-specific helper、drifted testを棚卸しする。

成果物:

```text
docs/productization/sim_rev3/codex_notes/phase_0_repo_audit.md
```

### Phase 0.5: Focused tests and baseline smoke stabilization

既存の最小focused checksを直す。新機能を入れない。

対象:

```text
build_comsol_case
source_preprocess / boundary_release
high_fidelity_collision / same-source skip
existing compare tools
```

### Phase 1: Minimal comparison rails

物理実装より前に、局所改善を誤採用しないためのcompare summaryを作る。

必要:

```text
import/preprocess status
post-first-step metrics
final state counts
wall interaction counts
first-crossing / vacuum time if available
runtime counters
```

### Phase 2: Mode separation + minimal manifest gate

`comsol_faithful` と `surface_release_production` を分ける。

### Phase 3: Import / coordinate / axisymmetric minimum semantics

coordinate scale、axis、field support、axisymmetric_rzを最小限固定する。

### Phase 4: Release canonicalization

capture tolerance と inward offset を分離し、release provenanceを保持する。

### Phase 5: Force breakdown / first-step parity

drag/electric/thermo/Brownianなどの寄与を切り分け、first-stepの差分を評価する。

### Phase 6: Release grace / wall event simplification

same-source skipをrelease直後の短いgraceに限定し、inward reimpactを通常wall eventへ戻す。

### Phase 7: Axisymmetric RZ completion

ring-area weighting、axis boundary semantics、source sampling measureを完成させる。

### Phase 8: V&V productization

小さなverificationケースとCOMSOL validationケースを整備する。

### Phase 9: Cleanup / simplification / performance

古い分岐、過剰diagnostics、使われないhelperを整理し、hot pathを最適化する。

## フェーズ順の理由

```text
評価なしに物理変更しない
テストdriftを放置して新設計を足さない
release/wallより前にfirst-stepを見られるようにする
速度改善は物理意味が固定されてから行う
```
