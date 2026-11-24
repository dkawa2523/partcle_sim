# Chamber Particles

外部の形状・場・粒子源から、粒子の運動・帯電・壁への到達を計算するPython基盤です。
現行の独立projectは `particle_platform_redesign/solver/` の **0.5.0 alpha** です。

[利用ガイド](particle_platform_redesign/solver/README.md)から、例題の生成、自分の入力の作成、検査・実行、
保存結果の解析・export・表示まで進めます。通常の操作は同じwheelのCLIとPythonで使えます。
COMSOLやソースcheckoutを通常実行の前提にしません。

- [文書案内](particle_platform_redesign/README.md)：現行仕様、設計、検証、未実装項目、実行記録への入口。
- [変更時の規約](AGENTS.md)：workspace境界とprojectの作業規則。
- [配布・変更状況](particle_platform_redesign/solver/RELEASE_NOTES.md)：実装版、Git/CI、公開の実施状況。

`model_dataset/` は外部入力とV&Vの参照資料、`old_code/` は旧実装のread-only archiveです。
どちらも通常のsolver依存先ではありません。過去の科学・性能・配布資格は当時のsourceと条件に限定し、
現行版の認定へ付け替えません。
