# 編集可能な配布例題

例題はwheel内の`chamber_particles_tools.examples`が定義し、通常入力と同じpublisherで作ります。
このdirectoryにcheckout専用のcase writerや別の設定一式は置きません。
例題の選び方、実行、CSV場への差替えと結果再利用は[利用ガイド](../README.md#例題を実行する)を参照してください。

```console
uv run --locked python -m chamber_particles init work/xy --example xy-ballistic
uv run --locked python -m chamber_particles check work/xy/case.yaml
```

生成先には`case.yaml`、`case.h5`、実現sourceの`particles.csv`、再作成用`recipe.yaml`が入ります。
`rz-charged`にはfield CSVとmapping、`reduced-electrostatic`にはthermal bundleとbuilder設定も含まれます。
CSV・単位・物性・モデル・時計を編集した後は、元の完成物を上書きせず別directoryへpublishします。
入力方法の詳細は[source producer](../src/chamber_particles_tools/source_builder/README.md)と
[CSV importer](../src/chamber_particles_tools/csv_import/README.md)が所有します。

これらは小型の解析的な利用例です。例題の粒径、source、step、壁則を別装置のdefaultにせず、
利用例の実行成功を実装置の予測妥当性へ読み替えません。
