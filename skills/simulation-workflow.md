# シミュレーションワークフロー

## 概要

EEZO×BtoB同時並行モデルのコスト妥当性を検証するためのシミュレーション手順書。

---

## Step 1: パラメータ読み込み・妥当性確認

### 実行コマンド
```python
from src.simulator import load_params_from_csv, SimulationParams

params = load_params_from_csv('data/raw/parameters.csv')
```

### 確認ポイント
- [ ] 金額単位が「円」で統一されているか
- [ ] 割合（率）が0〜1の範囲か
- [ ] 出典が明記されているか
- [ ] 極端な値がないか

### 出典確認チェックリスト
| パラメータ | 出典 | 確認状況 |
|-----------|------|----------|
| CVR 8% | 戦略レポート/ワイナリー実績 | ✓ |
| 顧客単価 8,000円 | 戦略レポート | ✓ |
| リピート率 25% | 戦略レポート | ✓ |
| インセンティブ 15% | 業界相場 | ✓ |

---

## Step 2: 各シナリオのコスト・売上計算

### シナリオA: BtoB営業単独

```python
result_a = simulator.simulate_btob_only()
```

**計算式**:
- 月間コスト = 固定費 + (訪問数 × 訪問単価) + (成約数 × 成約コスト)
- 月間新規顧客 = 訪問数 × 成約率
- 月間売上 = 累積顧客数 × 月間取引額

### シナリオB: EEZO単独

```python
result_b = simulator.simulate_eezo_only()
```

**計算式**:
- 月間コスト = プラットフォーム費 + 広告費 + (注文数 × 変動費) + (売上 × インセンティブ率)
- 月間新規顧客 = QRスキャン数 × CVR
- 月間売上 = 注文数 × 顧客単価

### シナリオC: 同時並行

```python
result_c = simulator.simulate_combined()
```

**計算式**:
- シナジー適用後の各パラメータで計算
- 追加コスト: 店舗連携費、コーディネーション費

---

## Step 3: KPI算出

### 主要KPI

| KPI | 計算式 | 判定基準 |
|-----|--------|----------|
| CAC | 総獲得コスト ÷ 新規顧客数 | 低いほど良い |
| LTV | 平均単価 × 購入頻度 × 継続期間 | 高いほど良い |
| LTV/CAC比率 | LTV ÷ CAC | **4以上が目標** |
| ROI | (売上 - コスト) ÷ コスト | 高いほど良い |
| BEP | 初期投資 ÷ 月間利益 | 短いほど良い |

### コード例

```python
results = simulator.run_all_scenarios()
df = simulator.results_to_dataframe(results)
print(df[['シナリオ', 'CAC（円）', 'LTV（円）', 'LTV/CAC比率']])
```

---

## Step 4: 感度分析

### 主要パラメータの感度分析

優先度の高いパラメータ:
1. `eezo_cvr` - EC CVR
2. `eezo_customer_price` - 顧客単価
3. `eezo_repeat_rate` - リピート率
4. `synergy_ltv_multiplier` - シナジー効果

### コード例

```python
sensitivity_df = simulator.sensitivity_analysis(
    param_name='eezo_cvr',
    variation_range=(0.7, 1.3),
    steps=11
)
```

### 分析観点
- どのパラメータが結果に最も影響するか
- 最悪ケースでも目標を達成できるか
- シナジー効果の前提は妥当か

---

## Step 5: 可視化

### 出力グラフ

1. **シナリオ比較バーチャート** (`outputs/scenario_comparison.png`)
2. **LTV/CAC比率比較** (`outputs/ltv_cac_comparison.png`)
3. **感度分析ヒートマップ** (`outputs/sensitivity_heatmap.png`)
4. **累積利益推移** (`outputs/cumulative_profit.png`)

### コード例

```python
from src.visualizer import (
    plot_scenario_comparison,
    plot_ltv_cac_comparison,
    plot_sensitivity_heatmap,
    plot_cumulative_profit
)

plot_scenario_comparison(df, save_path='outputs/scenario_comparison.png')
plot_ltv_cac_comparison(df, save_path='outputs/ltv_cac_comparison.png')
```

---

## Step 6: Markdownレポート生成

### 出力ファイル
`outputs/simulation_report.md`

### コード例

```python
from src.visualizer import generate_summary_report

report = generate_summary_report(df, output_path='outputs/simulation_report.md')
print(report)
```

### レポート構成
1. エグゼクティブサマリー
2. シナリオ比較表
3. 主要な発見
4. 次のアクション

---

## トラブルシューティング

### よくある問題

| 問題 | 原因 | 対処 |
|------|------|------|
| LTV/CACが異常に高い | LTV計算の継続期間が長すぎる | 現実的な値に修正 |
| 損益分岐点がNone | 月間利益がマイナス | パラメータを見直し |
| シナジー効果が過大 | 乗数が高すぎる | 保守的な値（1.1〜1.2）に修正 |

### パラメータ調整のガイドライン
- 初期分析: 楽観的な値は使わない
- 感度分析: ±30%の範囲で検証
- 意思決定: 悲観シナリオでも成立するか確認

---

## チェックリスト

### シミュレーション前
- [ ] パラメータ出典を確認
- [ ] 金額単位を統一
- [ ] 前提条件を文書化

### シミュレーション後
- [ ] LTV/CAC比率が4以上か確認
- [ ] 感度分析で脆弱なパラメータを特定
- [ ] レポートを生成・保存

### 報告前
- [ ] 計算ロジックをレビュー
- [ ] 前提条件の限界を明記
- [ ] 次のアクションを提案
