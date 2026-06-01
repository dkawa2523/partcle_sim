# MANIFEST

## docs/

- `00_original_goal_and_non_goals.md`  
  元の目的、最終的な製品像、やらないこと。

- `01_vigus_lessons_generalized.md`  
  VIGUSレポートから抽出した汎用的な失敗モード。

- `02_comsol_particle_tracing_reference_model.md`  
  COMSOL Particle Tracing と比較する際に意識すべきrelease、wall、time integration、axisymmetricの論点。

- `03_failure_modes_and_diagnostic_strategy.md`  
  差分原因を import / release / first-step / event / ensemble / runtime に分解する診断戦略。

- `04_revised_phase_roadmap.md`  
  v2で修正したフェーズ順。比較基盤を前倒しした理由を含む。

- `05_product_architecture_principles.md`  
  製品コードとして保守可能にするアーキテクチャ原則。

- `06_minimalism_policy.md`  
  過度な契約・診断・helper・testを避ける判断基準。

- `07_verification_validation_strategy.md`  
  小さな数値検証とCOMSOL validationを分けるV&V方針。

- `08_repository_placement.md`  
  この資料群、参照レポート、実装コード、出力成果物の配置指針。

- `09_risk_register.md`  
  Codex実装で起きやすい失敗と回避策。

## implementation_layers/

実装レイヤごとの責務、追加してよいもの、避けるべきもの、最小テスト。

## checklists/

各フェーズで作るもの、作らないもの、acceptance criteria。

## adrs/

小さなArchitecture Decision Records。実装時の判断根拠。

## templates/

manifest、compare summary、first-step CSV、cleanup inventory などのひな形。

## codex_notes/

Codexへ読ませる前提資料。プロンプト本文ではなく、実装時に守る前提です。
