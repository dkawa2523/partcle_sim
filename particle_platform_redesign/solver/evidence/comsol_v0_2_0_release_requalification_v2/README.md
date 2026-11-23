# 保存native参照の来歴

このdirectoryには、0.2.0調査時のnative観測・正規化・source/seed/単位の来歴を保持する。
現在のsolver能力、実行入口、精度資格を定義するdirectoryではない。

旧v46 candidateの実行recipe、設定、評価出力、図、publication用projectionは現行経路から撤去した。
それらの当時の内容と数値判断は[Git履歴](https://github.com/dkawa2523/partcle_sim/tree/75100e74f1243b527e065c9e6a68121221408b39/particle_platform_redesign/solver/evidence/comsol_v0_2_0_release_requalification_v2)に保存されている。
廃止したcampaign runnerや旧result readerを復活させて再実行する入口は提供しない。

残すものは次のnative参照に必要な記録である。

- `saved_native/`の観測binding、正規化、実行receiptとpath-only viewの来歴。
- `saved_native_*`と`publication_saved_native_closure.json`の、実観測・外部入力・元sourceを同定する台帳。
- C3に使ったprimitive、三電流の実現source、reference実行条件のreceipt。
- 元のnative policy/selection/seed authorityは、隣接する`comsol_binding_recert_2026_10_09_v1/`に残す。

nativeが実際に使用した`candidate_input_three_current_z0.h5`は、その名前にかかわらず実現sourceの外部入力である。
旧candidateの生成結果と同じ扱いで削除しない。原MPH、raw/normalized native CSV、実環境はこの撤去の対象ではない。

現行の明示入力による比較・準備は[外部V&V入口](../../tools/vv/comsol/README.md)を使う。
保存native記録のbindingは過去の参照条件を同定する証拠であり、現行candidateの数値収束や物理妥当性の証明ではない。
