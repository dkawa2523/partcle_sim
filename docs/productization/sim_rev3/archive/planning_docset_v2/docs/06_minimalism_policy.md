# 06. Minimalism Policy

このプロジェクトでは、保守性を守るために「足す」より「分ける・削る」を優先する。

## 追加してよいもの

- 比較で必要な小さなsummary。
- 既存の曖昧な責務を明確にする小さなdataclass。
- 1フェーズの挙動を守る最小テスト。
- 既存one-off診断を置き換える小さなCLI。
- 重複コードを減らすhelper。

## 避けるもの

- 大きなschema/contract framework。
- すべての層で同じvalidationを繰り返すこと。
- 何でも出すdiagnostic modeをdefaultにすること。
- private functionの細部に密着したテスト。
- VIGUS専用の条件分岐。
- `_tmp_*` 出力や過去probeを直接参照するコード。
- Codexのためだけに作る説明過多helper。

## Test policy

テストは3種類だけに寄せる。

```text
1. verification tests:
   小さな合成ケース。物理・数値の基本挙動を守る。

2. import/contract tests:
   入力意味論、coordinate, release, wall map を守る。

3. golden comparison smoke:
   小さな比較summaryが壊れていないことを見る。
```

避けるテスト:

- 大規模VIGUS fixtureをunit testにする。
- 出力メッセージ完全一致に過度依存する。
- helperの引数に密着して壊れやすい。
- 数値許容差の理由が不明。

## Diagnostics policy

通常run:

```text
run_summary.json
compare_summary.json
small counters
```

debug/deep run:

```text
wall_events.csv
force_contributions.csv
first_step_error.csv
trajectory_samples.csv
```

常時deep diagnosticsは禁止。
