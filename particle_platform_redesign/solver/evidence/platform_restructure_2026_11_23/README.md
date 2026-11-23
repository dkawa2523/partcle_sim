# 基盤再構成の作業記録

2026-11-23の再構成は`e9529fc024d2591d018b5809cd8c86a8f9b89f0f`に保存した。
通常clientのwheel統合、単一CLI、実行と共通のcheck、result4 context、engine/transport/executionの責務分離を行った。
当該sourceで1201 core/scenario tests、101 portable V&V tests、Windows/Linuxの配布確認が成功した。
これらはそのsourceに限定した履歴である。

役目を終えたsoftware確認log・古い性能出力・中間receiptは現行treeから撤去した。
当時の詳細はGit履歴に保持し、現在の手順・認定へ流用しない。
[現行整理の範囲と実行](../retirement_2026_11_23/README.md)を参照する。

[A/P各287粒子の科学的参照と実用比較](science/README.md)は維持した。
元MPH/native観測と候補40実現の同定、集団量・分母・局所差・乱数path一致を要求しない判断が対象である。
これは装置adhesion/flux、一般SDEやnative全方程式の認定ではない。
