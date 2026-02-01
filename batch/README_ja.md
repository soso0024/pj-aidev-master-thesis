# バッチテストケースジェネレーター

HumanEval/HumanEvalPackの複数の問題に対してテストケースジェネレーターの実行を自動化し、多くの問題のテストケースを一度に簡単に生成できるバッチ処理ツールです。

## 特徴

- **複数データセット**: HumanEvalとHumanEvalPackの両方をサポート (HuggingFace 🤗 からロード)
- **範囲ベースの生成**: HumanEval/0 から HumanEval/N までのテストを生成
- **特定のタスク選択**: カンマ区切りで特定のタスクIDを指定して生成
- **マルチモデル対応**: 複数のLLMモデルを同時に使用してテストを生成
- **進捗追跡**: リアルタイムの進捗更新と統計
- **エラー処理**: ユーザー制御による継続可能な堅牢なエラー処理
- **タイムアウト保護**: ハングアップを防ぐためのタスクごとのタイムアウト (5分)
- **包括的なレポート**: 詳細な成功/失敗の統計

## 使用方法

### 基本的な使用法

一連の問題に対するテストケースを生成します:

```bash
# 問題0-10のテストを生成 (両方のデータセットに対応する統一形式！)
python run_batch_test_case_generator.py --start 0 --end 10

# 範囲0-5に対してdocstringとASTを含めて生成
python run_batch_test_case_generator.py --start 0 --end 5 --include-docstring --include-ast

# 特定のタスクIDを指定 (統一番号形式 - 推奨)
python run_batch_test_case_generator.py --task-ids "0,5,10"

# HumanEvalPackデータセット (同じ形式！)
python run_batch_test_case_generator.py --start 0 --end 10 --dataset-type humanevalpack

# 範囲0-20に対して評価なしで高速生成
python run_batch_test_case_generator.py --start 0 --end 20 --disable-evaluation
```

### コマンドラインオプション

| オプション | 説明 | デフォルト |
| ---------- | ---- | ---------- |
| `--start N` | 開始タスクID番号 | 0 |
| `--end N` | 終了タスクID番号 | 50 |
| `--task-ids "X,Y,Z"` | カンマ区切りの特定タスクID | なし (範囲を使用) |
| `--models MODEL1 MODEL2` | 使用するLLMモデル (複数指定可) | gemini-2.5-flash |
| `--dataset PATH` | レガシーパラメータ (データセットはHuggingFace 🤗からロード) | N/A |
| `--dataset-type TYPE` | 使用するデータセット (humaneval または humanevalpack) | humaneval |
| `--output-dir DIR` | テストファイルの出力ディレクトリ | generated_tests |
| `--include-docstring` | プロンプトに関数docstringを含める | False |
| `--include-ast` | プロンプトに正解のASTを含める | False |
| `--disable-evaluation` | テストの自動評価をスキップ | False |
| `--quiet-evaluation` | 評価出力を簡潔にする | False |
| `--max-fix-attempts N` | タスクごとの最大修正試行回数 | 3 |
| `--task-timeout N` | 各タスクのタイムアウト (秒) | 300 (5分) |

### マルチモデル生成

複数のLLMモデルを同時に使用してテストを生成します:

```bash
# 包括的なテストのために複数のモデルを使用
python run_batch_test_case_generator.py --start 0 --end 5 --models claude-sonnet-4-5 gemini-2.5-flash

# 範囲にわたってモデルのパフォーマンスを比較
python run_batch_test_case_generator.py --start 0 --end 10 --models claude-sonnet-4-5 gpt-5.1
```

## 例

### フルコンテキストで問題0-10のテストを生成:

```bash
python run_batch_test_case_generator.py --start 0 --end 10 --include-docstring --include-ast
```

### 評価を無効にして特定の問題を生成:

```bash
# 統一番号形式 (両方のデータセットで動作！)
python run_batch_test_case_generator.py --task-ids "0,15,30" --disable-evaluation

# HumanEvalPack
python run_batch_test_case_generator.py --task-ids "0,15,30" --dataset-type humanevalpack --disable-evaluation
```

### 自動化のための静かなバッチ処理:

```bash
python run_batch_test_case_generator.py --start 0 --end 50 --quiet-evaluation --max-fix-attempts 1
```

### 複雑な問題のためのカスタムタイムアウト:

```bash
python run_batch_test_case_generator.py --start 0 --end 10 --task-timeout 600 --include-docstring
```

## インタラクティブ機能

### エラー処理

バッチ処理中にタスクが失敗した場合、次のようにプロンプトが表示されます:

```
❓ Task HumanEval/5 failed. Continue with remaining tasks? (y/n/q):
```

オプション:

- `y` (yes): 次のタスクに進む
- `n` (no): バッチ処理を停止
- `q` (quit): プログラムを直ちに終了

**注意**: `--quiet-evaluation` を使用する場合、失敗したタスクは自動的にスキップされ、次のタスクに進むため、自動化に適しています。

### 進捗追跡

リアルタイムの進捗更新:

```
📊 Progress: 3/10 (30.0%)
🚀 Processing HumanEval/2
```

### 最終サマリー

包括的なバッチ処理レポート:

```
🏁 BATCH PROCESSING COMPLETE
📊 Summary:
  Total tasks: 10
  ✅ Successful: 8
  ❌ Failed: 2
  ⏭️  Skipped: 0
  ⏱️  Duration: 145.3 seconds
  📁 Output directory: generated_tests
```

## 出力構造

バッチジェネレーターは、タスクIDごとに整理された、単一ジェネレーターと同じファイル構造を作成します:

```
generated_tests/
   test_humaneval_0_success.py
   test_humaneval_0_success.stats.json
   test_humaneval_1_docstring_false.py
   test_humaneval_1_docstring_false.stats.json
   ...
```

## パフォーマンスに関する考慮事項

- **タイムアウト**: ハングアップを防ぐため、各タスクには設定可能なタイムアウト (デフォルト5分) があります
- **メモリ**: メモリ使用量を管理するため、プロセスは順次実行されます
- **API制限**: Claude APIのレート制限を自動的に尊重します
- **ディスク容量**: 大規模なバッチ生成を行う場合は、空き容量を監視してください

### タイムアウト設定

`--task-timeout` を使用して、必要に応じてタスクごとのタイムアウトを調整します:

- **単純な問題**: `--task-timeout 180` (3分)
- **複雑な問題**: `--task-timeout 600` (10分)
- **非常に複雑な問題**: `--task-timeout 900` (15分)

## エラー回復

バッチジェネレーターには堅牢なエラー処理が含まれています:

1. **サブプロセスエラー**: コマンドの失敗をキャプチャして報告
2. **タイムアウト保護**: 問題のあるタスクでの無限ハングアップを防止
3. **ユーザー制御**: 失敗時の継続または停止を選択可能
4. **グレースフルシャットダウン**: Ctrl+Cによる中断をきれいに処理

## メインツールとの統合

バッチジェネレーターは、メインの `run_test_case_generator.py` ツールをラップします。ハングアップを避けるため、常に `--no-show-prompt` を強制して非対話モードで実行します。インタラクティブなプロンプトプレビューが必要な場合は、単一実行ツールを直接使用してください。

## 要件

- Python 3.10+
- メインテストジェネレーターのすべての依存関係
- 同じディレクトリ内の `run_test_case_generator.py`
- 初回実行時に **インターネット接続** が必要 (HuggingFace 🤗 からデータセットをダウンロードするため)
- データセットは初回ダウンロード後に自動的にキャッシュされます

## 効果的なバッチ処理のためのヒント

1. **小さく始める**: まずは狭い範囲でテストする (例: --start 0 --end 5)
2. **静音モードを使用**: 大規模なバッチには `--quiet-evaluation` を追加
3. **進捗を監視**: 成功/失敗のパターンに注意
4. **ディスク容量を確認**: 大規模なバッチは多数のファイルを生成します
5. **コストを考慮**: フルオプションでのバッチ処理は高額になる可能性があります
