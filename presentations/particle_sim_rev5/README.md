# 粒子輸送計算の技術説明資料

[編集可能なPowerPoint（26枚）](particle_sim_rev5_source_matched.pptx)

対象: `sim_rev5`、内容確認時のコミット `fa51c2e10f32153afbb08d54012ec676f70508c4`。

- 1–14枚: 用途、コード構成、入力・出力、計算ワークフロー、物理モデル・方程式、検証範囲。
- 15–26枚: 元画像12枚の構成を踏まえた編集可能な概念図。図形・線・文字を個別編集できます。グラフ2点は個別画像です。
- 元画像は画像生成による概念説明用です。計算結果や実測データではありません。元画像中の表現と実装が異なる箇所は、PowerPointの説明・注記を参照してください。

## 元画像とスライドの対応

| 図番号 | スライド | 元画像 |
|---|---|---|
| 01 | 15 | [外部場を使った粒子輸送](original_figures/01_particle_transport.png) |
| 02 | 16 | [中性気体抗力](original_figures/02_neutral_drag.png) |
| 03 | 17 | [OML帯電と運動の連成](original_figures/03_oml_coupling.png) |
| 04 | 18 | [集約二電流・三電流](original_figures/04_effective_charging.png) |
| 05 | 19 | [イオン抗力](original_figures/05_ion_drag.png) |
| 06 | 20 | [追加の力](original_figures/06_additional_forces.png) |
| 07 | 21 | [場の補間](original_figures/07_field_interpolation.png) |
| 08 | 22 | [時間積分](original_figures/08_time_integration.png) |
| 09 | 23 | [最初の壁面接触](original_figures/09_first_contact.png) |
| 10 | 24 | [壁面応答とRZ軸](original_figures/10_boundary_response.png) |
| 11 | 25 | [熱揺らぎを含む粒子輸送](original_figures/11_brownian_transport.png) |
| 12 | 26 | [実行処理と検証](original_figures/12_execution_validation.png) |
