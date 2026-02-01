# HumanEval テストケースジェネレーター

複数のLLMプロバイダーを使用して、HumanEvalおよびHumanEvalPackの問題に対する包括的なpytestテストケースを自動生成し、評価、エラー修正、詳細な分析を行うツールです。

## 特徴

- **マルチモデル対応**: Claude (Opus, Sonnet, Haiku), Gemini, GPTモデル
- **複数データセット**: HumanEvalおよびHumanEvalPackのサポート (HuggingFace 🤗 からロード)
- **自動評価**: LLMによるエラー修正機能を備えたPytest実行
- **バッチ処理**: 複数の問題に対するテストを一括生成
- **プロンプトエンジニアリング**: 比較のための4つの異なるプロンプト戦略
- **包括的な分析**: モデル比較を含むデータセット認識型の可視化
- **コスト追跡**: トークン使用量とAPIコストのモニタリング
- **カバレッジ分析**: テストカバレッジ率の追跡

## クイックスタート

1. 依存関係のインストール: `uv sync` または `pip install -r requirements.txt`
2. APIキーの設定: `export ANTHROPIC_API_KEY="your-key"` (またはGemini用の `GOOGLE_API_KEY`, GPT用の `OPENAI_API_KEY`)
3. テスト生成: `python run_test_case_generator.py` (**Python 3.10+ が必要**)

**注意**: データセット (HumanEval および HumanEvalPack) は、初回実行時に HuggingFace 🤗 から自動的にダウンロードされます。手動でのセットアップは不要です！

## サポートされているモデル

`models_config.json` で設定されているモデル:

### Claude モデル (Anthropic)

- **Claude Opus 4.5** - 最も高性能、最高コスト
- **Claude Sonnet 4.5** - 知能と速度のベストバランス
- **Claude Haiku 4.5** - 高速かつ高性能

### Gemini モデル (Google)

- **Gemini 3 Pro Preview** - 最新のプレビューモデル
- **Gemini 2.5 Flash** - 高速かつ高性能 (一部のティアで無料)

### GPT モデル (OpenAI)

- **GPT-5.2** - 高い知能
- **GPT-5.1** - バランス型
- **GPT-5 Mini** - 小型かつ高速
- **GPT-5 Nano** - 極めて高速かつ軽量

**デフォルトモデル**: `gemini-2.5-flash`

## 使用方法

### 単一テスト生成

```bash
# デフォルトモデルでランダムな問題 (HumanEval)
python run_test_case_generator.py

# 特定のモデルで特定の問題 (統一ID形式 - 推奨)
python run_test_case_generator.py --task-id 0 --models claude-sonnet-4-5

# Geminiモデルを使用
python run_test_case_generator.py --task-id 5 --models gemini-2.5-flash

# フルコンテキスト生成 (docstringとASTを含める)
python run_test_case_generator.py --task-id 10 --include-docstring --include-ast

# HumanEvalPackデータセットを使用 (同じ統一ID形式！)
python run_test_case_generator.py --dataset-type humanevalpack --task-id 0

# レガシー形式もサポート
python run_test_case_generator.py --task-id "HumanEval/0"
python run_test_case_generator.py --dataset-type humanevalpack --task-id "Python/0"
```

### バッチ処理

```bash
# 問題0-10のテストを生成 (HumanEvalとHumanEvalPackの両方で動作！)
cd batch
python run_batch_test_case_generator.py --start 0 --end 10

# 特定のモデルを使用
python run_batch_test_case_generator.py --start 0 --end 10 --models claude-haiku-4-5

# HumanEvalPackデータセット (同じコマンド形式！)
python run_batch_test_case_generator.py --start 0 --end 10 --dataset-type humanevalpack

# 特定のタスクID (統一形式)
python run_batch_test_case_generator.py --task-ids "0,5,10,15"

# 比較のために複数のモデルを使用
python run_batch_test_case_generator.py --start 0 --end 5 --models claude-sonnet-4-5 gemini-2.5-flash

# その他のオプションについては batch/README.md を参照
```

### プロンプトエンジニアリング比較

```bash
# 異なるプロンプト戦略を比較
python prompt_engineering_comparison.py --task-id "HumanEval/0"
```

### 分析と可視化

```bash
# 分析プロットを生成 (Python 3.8.20+ が必要)
python run_analysis.py

# 特定の結果ディレクトリを指定
python run_analysis.py --results-dir data/generated_tests_claude-haiku-4-5/ --output-dir vizs/
```

`vizs/` フォルダに可視化を作成します:

- 成功率とカバレッジ分析
- コスト対パフォーマンス指標
- アルゴリズムの複雑さ分析
- データセット認識型の問題分類
- **効率性指標の比較** (CCE-C0, CCE-C1, SCCE)

## 効率性指標

分析には、コストパフォーマンスのトレードオフを評価するための独自の効率性指標が含まれています。

| 指標 | 式 | 説明 |
| ---- | -- | ---- |
| **CCE-C0** | C0 Coverage / (Cost × 1000) | 命令網羅率の効率性 |
| **CCE-C1** | C1 Coverage / (Cost × 1000) | 分岐網羅率の効率性 |
| **SCCE** | Success × (0.3×C0 + 0.7×C1) / (Cost × 1000) | 成功率で重み付けされた網羅率の効率性 |

### 指標の解釈

- **CCE-C0 / CCE-C1**: 値が高いほどコスト効率が良いことを示します。値が10.0の場合、$0.001あたり10%のカバレッジを達成したことを意味します。
- **SCCE**: 成功率と重み付けされたカバレッジを組み合わせたものです。成功したテストケースのみがスコアに寄与するため、最も包括的な指標です。
- **重み付けカバレッジ**: `0.3×C0 + 0.7×C1` - C1 (分岐網羅) を達成すれば C0 (命令網羅) も達成される (C1 ⊃ C0) ため、C1の方を高く重み付けしています。

### 解釈の例

| モデル | CCE-C0 | CCE-C1 | SCCE | 解釈 |
| ------ | ------ | ------ | ---- | ---- |
| Claude Haiku | 7.1 | 8.0 | 7.73 | 高効率、予算が限られている場合に適している |
| Claude Sonnet | 3.2 | 3.5 | 3.4 | 効率は低いが、絶対的なカバレッジは高い可能性がある |

> **注意**: SCCEをモデル間で比較して、ユースケースに最適なコストパフォーマンスのトレードオフを見つけてください。


## プロジェクト構造

```
├── analysis/                           # 分析モジュール
│   ├── analysis_reporter.py           # レポート生成
│   ├── cross_model_plots.py           # モデル比較プロット
│   ├── data_loader.py                 # データロードユーティリティ
│   ├── dataset_aware_plots.py         # データセット固有の可視化
│   ├── humanevalpack_plots.py         # HumanEvalPack固有の可視化
│   ├── problem_classifier.py          # 問題分類ロジック
│   └── traditional_plots.py           # 一般的な可視化
├── batch/                             # バッチ処理
│   ├── README.md                      # バッチ処理のドキュメント
│   └── run_batch_test_case_generator.py
├── data/                              # 生成されたテスト出力
│   └── generated_tests_[dataset]_[model]/
├── evaluator/                         # テスト評価ロジック
├── generator/                         # テストケース生成ロジック
├── llm_clients/                       # LLMクライアント実装
├── problem_classification/            # 詳細な分類データ
├── prompts/                           # プロンプトテンプレート
│   ├── basic.txt                     # 基本プロンプト
│   ├── docstring.txt                 # docstring付き
│   ├── ast.txt                       # AST付き
│   ├── docstring_ast.txt             # フルコンテキスト
│   └── README.md                     # プロンプトドキュメント
├── utils/                             # 詳細なユーティリティ
├── vizs/                             # 分析可視化
├── config.py                         # 設定ファイル
├── model_utils.py                    # モデルユーティリティ関数
├── prompt_engineering_comparison.py   # プロンプト戦略比較
├── remove_duplicates.py              # 重複ファイル削除ユーティリティ
├── run_test_case_generator.py        # メインスクリプト
├── run_analysis.py                   # 可視化生成
├── run_cross_model_analysis.py       # クロスモデル分析スクリプト
├── models_config.json                # モデル設定
└── pyproject.toml                    # プロジェクト依存関係
```

## ファイル出力

- **テストファイル**: `test_python_X_[config]_[status].py`
- **統計情報**: `test_python_X_[config]_[status].stats.json`
- **可視化**: `vizs/` 内の分析プロット
- **プロンプト結果**: `prompt_comparison_results/` 内の比較データ

## テストの実行

```bash
# 生成されたテストを実行
cd data/generated_tests_[dataset]_[model]
pytest test_python_0_*.py -v --cov

# 特定のテストを実行
pytest test_python_0_missing_logic_success.py -v
```

## コストガイド

| モデル | 入力/1K | 出力/1K | 用途 |
| ------ | ------- | ------- | ---- |
| Claude Opus 4.5 | $0.005 | $0.025 | 複雑な問題 |
| Claude Sonnet 4.5 | $0.003 | $0.015 | ベストバランス |
| Claude Haiku 4.5 | $0.001 | $0.005 | 高速、有能 |
| Gemini 3 Pro Preview | $0.002 | $0.012 | 高性能 |
| Gemini 2.5 Flash | Free* | Free* | 高速 & 効率的 |
| GPT-5.2 | $0.00175 | $0.014 | 高い知能 |
| GPT-5.1 | $0.00125 | $0.010 | バランス型 |
| GPT-5 Mini | $0.00025 | $0.002 | 小型 & 高速 |
| GPT-5 Nano | $0.00005 | $0.0004 | 極めて軽量 |

\*一部のティア / プレビュー期間中は無料

## 要件

- テスト生成には **Python 3.10+** が必要 (`run_test_case_generator.py`)
- 分析スクリプトには **Python 3.8.20+** が必要 (`run_analysis.py`)
- `uv sync` または `pip install -r requirements.txt`
- 初回実行時に **インターネット接続** が必要 (HuggingFace 🤗 からデータセットをダウンロードするため)
- APIキー:
  - Claudeモデル用 `ANTHROPIC_API_KEY`
  - Geminiモデル用 `GOOGLE_API_KEY`
  - GPTモデル用 `OPENAI_API_KEY`

> **注意**:
>
> - データセットは初回ダウンロード後に自動的にキャッシュされます
> - Pythonのバージョン要件が異なるため、テスト生成 (3.10+) と分析 (3.8.20+) で別々の仮想環境が必要になる場合があります

## 環境変数

```bash
# Claudeモデル用
export ANTHROPIC_API_KEY="your-anthropic-key"

# Geminiモデル用
export GOOGLE_API_KEY="your-google-key"

# GPTモデル用
export OPENAI_API_KEY="your-openai-key"
```

## プロンプト戦略

`prompts/` で4つのプロンプト戦略が利用可能です:

1. **basic.txt** - 最小限のコンテキスト、関数シグネチャのみ
2. **docstring.txt** - 関数のdocstringを含む
3. **ast.txt** - 正解のASTを含む
4. **docstring_ast.txt** - フルコンテキスト (docstring + AST)

`--include-docstring` および `--include-ast` フラグを使用して戦略を選択してください。
