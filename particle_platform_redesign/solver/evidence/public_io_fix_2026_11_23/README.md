# 公開入力・出力の不具合修復

開始source: `1ab0ee9e3f94d169be198929ef408c7e7b90325c`。対象は最終レビューのR06/R07/R15/R16と、その入力・変換・公開に直接関係する必須影響先。
物理model、積分、event、RNG、結果schema、COMSOL解法の全面改修は含めない。

## 原因と修復

| 対象 | 確認した原因 | 修復した責務と受渡し |
|---|---|---|
| R15 selected CSV | 全列をfloatへ変換した後では、隣接した大整数IDを区別できない | 元numeric tokenからscopeとexact int64 node IDを解決。選択後IDと物理floatを分離し、同じpermutationでjoinする |
| R06 source入力 | 設定検査より先にfilterを消費し、CSV余剰cellを落とす。数値filterにもfloat丸めがある | typed source specificationを先に解決。header/全row幅→filter→選択行の値・IDの順へ。numeric filterはexact decimal比較 |
| R16 static builder | canonicalとして合法な時系列fieldを、producerが静的配列と解釈した | check/build共通prepareがStaticThermalInputを確定し、時系列thermalはsolve前に拒否。consumerのFieldData再解釈を除去 |
| R07 analyze公開 | 完成JSONより先に公開先fileを作る | serialize→同directoryのtemporaryへwrite/flush/fsync→hard-linkで新規公開。既存先を上書きしない |

R15は、別の外部COMSOL readerの修正とは別ownerに残っていた。以前の調査・採点はこれら4項目を未実装と明記しており、修復済みcodeが再発したとの証拠ではない。
これまでの正常例では、隣接大整数、時系列thermal、不正nested設定、書込み中断を確認できていなかった。
今回の回帰はその原因となる境界条件を公開Python/CLIから通す。

## 関連する必須修正

- source numeric filterが`2**53+1`を`2**53`として選ぶことを実再現し、raw decimalと解決済み期待値の比較へ修復した。
- canonical範囲外のdomain等が配列作成でOverflowErrorとなることを実再現し、既存integer decoderでint32/int64の範囲を解決する。
- 巨大basis入力のOverflowErrorと、極端な単位変換のRuntimeWarningが機械可読stderrを壊す問題を、入力・数値変換のownerで扱う。
- JSON公開後のtemporary削除だけが失敗した場合は、完成targetをrollbackせず、公開済みと後処理失敗を明示する。公開前失敗と区別する。

例外をCLIで一律に握りつぶしたり、非有限値をdefaultで置き換えたりしない。
単位・数値変換の失敗は有限性を確認して拒否し、通常値の式・物性・ID・weightを維持する。

## 検証方針

既存のCSV affine/axis/単位/coordinate参照、sourceの速度・表面・inheritance、static電場解析解、公開結果集計を保持する。
新しい確認は同じtest層へ追加し、常設の診断・contract・migration frameworkを作らない。
adapter revisionはv5、builder revisionはv4。入力設定形式とcanonical/result/checkpoint schema、数値coreの式は変更しない。
既存の正しく解決できた値は保持するが、producer revision/provenanceの更新によりcanonical content hashは変わり得る。

## 実行結果と完了範囲

独立レビュー4件と全体closeoutの必須指摘は解消済み。[実行・完了記録](execution.json)にcommand、再現、正常操作とsource bindingを集約した。

| 最終確認 | 実行結果 |
|---|---|
| 全verification/scenario | 1298 passed /619.77 s。errors/failures/skipped 0 |
| 外部workflow（COMSOL不要） | 156 passed /51.43 s。errors/failures/skipped 0 |
| 品質 | lock、Ruff185 files、Pyrefly0 errors、依存6契約、CC<16、文書33/skills4が成功 |
| 実行source | ソース・テスト等176 filesのhashが最終実行開始前と完了後で一致。core productionの変更なし |
| 公開正常操作 | 自分のgeneric電場CSV→exact ID import→自分のsource/recipe→check/run→inspect/analyze/export/view、全9 CLI成功 |
| 独立解析例 | 1粒子、a=eZE/m。壁到達誤差1.73e-17 s、Decimal60での保存5点の最大位置差6.94e-18 m、failure0 |
| 再利用 | particle ID・mass・charge・weight・odd mesh IDを保持し、集計/export/表示のsource identity一致 |

最終全体testのwarning1件は、既存の極端なinvalid scalar-budget拒否testによるevents.pyのoverflow観測。
今回のsource変換warningによる機械可読JSON破損は、公開再現を再実行して解消を確認した。
中間1291 passは最後のpeer修正前に開始した別runで、最終1298の代わりに使用していない。
最初の外部testはconsole版pytestの呼出しによりcollection errorとなり、文書どおり`uv run --locked python -m pytest`へ訂正した。import回避のproduction変更はしていない。

再発確認の入口:

```console
uv run --locked python -m pytest tests/verification/test_csv_adapter.py tests/verification/test_source_builder.py tests/verification/test_electrostatic_builder.py tests/scenarios/test_source_input_errors.py tests/scenarios/test_builder_static_inputs.py tests/scenarios/test_analysis_publication.py -q
uv run --locked python -m pytest tests/verification tests/scenarios -q
uv run --locked python -m pytest tools/vv/comsol/tests tools/vv/gas_collision/tests -q
```

必要な旧inputの再import、旧不完全JSONの扱いは各client READMEへ追記した。
AGENTS.mdの安定した責務・方針は維持し、一時probeの実行器や全体gateを包む独自runnerは追加していない。
原MPH、native/実験data、旧archive、既存環境はこの修復で変更していない。
COMSOL再計算、大規模性能matrix、clean wheel再install、可視reportの新規buildは未実施。数値core・依存・配布配置・reportデータを変えていないため今回の4修復の完了条件には追加しない。
新しい物理精度・装置予測・全native parityの認定は本記録の主張に含めない。
