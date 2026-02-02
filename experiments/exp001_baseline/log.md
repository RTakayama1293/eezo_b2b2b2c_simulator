# 実験ログ: exp001_baseline

## 実験概要

| 項目 | 内容 |
|------|------|
| 実験ID | exp001 |
| 目的 | 3シナリオの基準値シミュレーション |
| 実行日 | [TODO: YYYY-MM-DD] |
| 担当者 | [TODO] |

---

## パラメータ設定

### BtoB営業
- 月間訪問数: 20件
- 成約率: 10%
- 年間取引額: 120万円

### EEZO（EC）
- 連携店舗数: 10店舗
- CVR: 8%
- 顧客単価: 8,000円
- リピート率: 25%

### シナジー効果
- BtoB CVR向上: ×1.2
- EEZO CVR向上: ×1.3
- LTV向上: ×1.25
- コスト効率化: ×0.85

---

## 実行コマンド

```python
from src.simulator import SimulationParams, CostSimulator
from src.visualizer import plot_scenario_comparison, generate_summary_report

# シミュレーション実行
params = SimulationParams()
simulator = CostSimulator(params)
results = simulator.run_all_scenarios()

# 結果出力
df = simulator.results_to_dataframe(results)
print(df.to_string())

# 可視化
plot_scenario_comparison(df, save_path='outputs/scenario_comparison.png')

# レポート生成
generate_summary_report(df, output_path='outputs/simulation_report.md')
```

---

## 結果

### シナリオ比較

| シナリオ | LTV/CAC | 月間利益 | 3年ROI |
|----------|---------|----------|--------|
| A. BtoB単独 | [TODO] | [TODO]万円 | [TODO]% |
| B. EEZO単独 | [TODO] | [TODO]万円 | [TODO]% |
| C. 同時並行 | [TODO] | [TODO]万円 | [TODO]% |

### 主要な発見

1. [TODO: 発見1]
2. [TODO: 発見2]
3. [TODO: 発見3]

---

## 考察

### 仮説との比較
- [TODO]

### 想定外の結果
- [TODO]

### 追加検証が必要な点
- [TODO]

---

## 次のアクション

- [ ] [TODO: アクション1]
- [ ] [TODO: アクション2]
- [ ] [TODO: アクション3]

---

## 出力ファイル

- `outputs/scenario_comparison.png` - シナリオ比較グラフ
- `outputs/simulation_report.md` - サマリーレポート
- `data/processed/results_exp001.csv` - 詳細結果データ
