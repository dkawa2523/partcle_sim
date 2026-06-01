# Layer 01: Import and Case Build

## 目的

geometry、field、release、boundary mapを同じ座標系・単位系に揃える。

## 重要責務

- coordinate scaleを明示する。
- coordinate systemを明示する。
- field bundle axesをgeometryに反映する。
- boundary mapとpart mapを明示する。
- valid maskをsolver側で勝手に拡張しない。
- hard invalid と mixed stencil を区別する。

## 最小成果物

```text
case_import_summary.json
provider_contract_report.json
input_contract_report.json
```

## 避けること

- geometryとfieldを別々の暗黙axisで読む。
- COMSOL .mphをsolver coreで読む。
- missing fieldをsolverが勝手に補間・補修する。
- case builderでVIGUS固有の壁IDを前提にする。
