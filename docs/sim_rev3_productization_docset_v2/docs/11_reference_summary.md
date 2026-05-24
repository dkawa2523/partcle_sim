# 11. Reference Summary

## VIGUS report lessons used

この資料群は、VIGUSレポートから以下を前提としている。

- 初期差分の主因はsame-source skip単独ではない。
- 最初に効いたのはimport / case build / initial condition / boundary release / field-force parity。
- specular補正とsame-source tighteningは後段の局所改善。
- focused checksにはdriftが残っていたため、現状をgreenと仮定して新設計を積まない。
- `preprocess_ratio_p50` が良くても `first_step_ratio_p50` が悪い場合、first-step力学を独立に見る必要がある。
- wall104のような局所branchは診断として有用だが、global fixとは限らない。
- same-source skipはruntime改善だけでなく物理境界bypassの危険を持つ。
- sampledとfull releaseの両方が必要。

## COMSOL reference model lessons used

- Particle Tracingのrelease、wall、force、time steppingは一体のモデルとして比較する。
- Release from Data File は初期位置・速度・補助変数をファイルから指定できる。
- Wall conditionにはfreeze, bounce, stick, disappear, diffuse, mixed diffuse/specularなど複数の物理条件がある。
- first-order Newtonian formulation と second-order Newtonian formulationでは既定time steppingが異なる。
- 2D axisymmetricではAxial Symmetry nodeが特別な意味を持つ。
