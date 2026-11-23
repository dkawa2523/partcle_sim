# Chamber Particles

外部の形状・場・粒子源から、粒子の運動・帯電・境界イベントを計算するPython基盤です。
現行の開発版は **0.5.0 alpha**。CLIとPythonで入力作成、検査、計算、結果の解析・表示まで完結し、
通常機能は同じwheelに含まれます。COMSOLとソースcheckoutは通常実行の前提ではありません。

[利用手順と実行例](particle_platform_redesign/solver/README.md)から始めてください。
`init`で編集可能な例題を作り、`create`で自分のcanonical入力とCSV粒子源を組み合わせます。
`check`は実行と同じ準備を行い、`run`が一つの計算engineで実行します。
`inspect`、`analyze`、`export`、`view`は保存結果だけを使います。

| 場所 | 役割 |
|---|---|
| [particle_platform_redesign/](particle_platform_redesign/README.md) | 現行仕様、独立uv project、実装、開発・科学検証の記録 |
| `model_dataset/` | 入力provenanceと外部V&Vの参照資料。solverの依存先ではありません |
| `old_code/` | former implementationのread-only archive。環境・packageを再利用しません |

計算coreは`chamber_particles`、通常clientと唯一のCLIは`chamber_particles_tools`が所有します。
live COMSOLの抽出や開発V&Vは外部toolです。COMSOLとの実用的な対応、数値的一致の認定、
実装置の予測妥当性は別の主張として扱います。

入力case/dataはschema 3、保存結果は自己完結した解釈contextを持つschema 5、checkpointはlayout 2です。
旧結果を無言で読む互換経路はありません。過去のRelease・科学計算・性能資格は当時の記録に固定し、
現行版の認定へ付け替えません。0.5.0の公開・配布状況は[release notes](particle_platform_redesign/solver/RELEASE_NOTES.md)と
実際の配布記録で確認してください。

変更前に[AGENTS.md](AGENTS.md)と[現行領域の規約](particle_platform_redesign/AGENTS.md)を読んでください。
