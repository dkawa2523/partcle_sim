# Numerical verification tests

解析解、manufactured solution、収束、event geometryの検証だけをここへ置きます。P02では
`microcases.py`がC01～C10のcanonical入力と独立oracleを一時directoryへ生成します。production packageは
このtest moduleや`expected.json`をimportせず、各数値能力のowner packageが実装された時点で比較を
有効化します。解析値の記録精度とmethod固有の受入差は別keyです。event oracleの`boundary_events`は
完全logそのものではなく、boundary ordinalの振り直し、candidate list展開、law意味名への解決を行った
比較projectionです。resolved lawの列名は標準event logと同じ`law_id`です。release/failure行を含む
完全logではありません。COMSOL結果、旧solver、private helper
配置を正解にしません。
