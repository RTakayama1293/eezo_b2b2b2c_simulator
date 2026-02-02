"""
EEZO×BtoB 同時並行モデル 包括的シミュレーション分析

分析内容:
1. 3シナリオの比較（BtoB単独、EEZO単独、同時並行）
2. 主要KPIの算出（CAC, LTV, ROI, BEP）
3. 多変量感度分析
4. コスト構造分析
5. NPV/IRR分析
6. 時系列収益シミュレーション
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.simulator import SimulationParams, CostSimulator, load_params_from_csv, ScenarioResult
from src.visualizer import (
    plot_scenario_comparison,
    plot_ltv_cac_comparison,
    plot_cumulative_profit,
    plot_sensitivity_heatmap,
    generate_summary_report,
    set_style
)
from src.utils import setup_logger, format_currency, calculate_npv, calculate_irr

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

logger = setup_logger('simulation')


def analyze_cost_breakdown(params: SimulationParams) -> Dict[str, Dict[str, float]]:
    """
    各シナリオのコスト内訳を詳細分析

    Returns:
        シナリオ別のコスト内訳辞書
    """
    p = params

    # シナリオA: BtoB営業単独
    btob_fixed = p.btob_monthly_salary + 25_000  # 人件費+諸経費
    btob_visit_variable = p.btob_monthly_visits * p.btob_visit_cost
    btob_deal_variable = p.btob_monthly_visits * p.btob_conversion_rate * p.btob_deal_cost

    btob_costs = {
        '人件費': p.btob_monthly_salary,
        '諸経費（通信・ツール）': 25_000,
        '訪問変動費': btob_visit_variable,
        '成約変動費': btob_deal_variable
    }

    # シナリオB: EEZO単独
    total_qr_scans = p.eezo_partner_stores * p.eezo_qr_scans_per_store
    monthly_new_customers = total_qr_scans * p.eezo_cvr
    monthly_orders = monthly_new_customers * (1 + p.eezo_repeat_rate * (p.eezo_purchase_frequency - 1) / 12)
    monthly_revenue = monthly_orders * p.eezo_customer_price

    eezo_costs = {
        'プラットフォーム費': p.eezo_platform_fee,
        '広告費': p.eezo_ad_budget,
        '配送費': monthly_orders * p.eezo_shipping_cost,
        '梱包費': monthly_orders * p.eezo_packing_cost,
        '店舗インセンティブ': monthly_revenue * p.eezo_commission_rate
    }

    # シナリオC: 同時並行
    combined_costs = {
        'BtoB固定費（効率化後）': (btob_fixed + btob_visit_variable + btob_deal_variable) * p.synergy_cac_efficiency,
        'EEZO運営費（効率化後）': (p.eezo_platform_fee + p.eezo_ad_budget * 0.5) * p.synergy_cac_efficiency,
        'EEZO変動費': monthly_orders * (p.eezo_shipping_cost + p.eezo_packing_cost),
        '店舗インセンティブ': monthly_revenue * p.eezo_commission_rate,
        '店舗連携費': p.store_monthly_incentive * p.eezo_partner_stores,
        'コーディネーション費': p.coordination_cost
    }

    return {
        'A. BtoB営業単独': btob_costs,
        'B. EEZO単独': eezo_costs,
        'C. BtoB×EEZO同時並行': combined_costs
    }


def analyze_monthly_progression(
    results: List[ScenarioResult],
    params: SimulationParams,
    months: int = 36
) -> pd.DataFrame:
    """
    月次の収益・コスト推移を分析

    Returns:
        月次データのDataFrame
    """
    data = []

    for result in results:
        cumulative_profit = 0
        initial_investment = 0

        # 同時並行モデルは初期投資あり
        if result.name == "C. BtoB×EEZO同時並行":
            initial_investment = params.store_setup_cost * params.eezo_partner_stores
            cumulative_profit = -initial_investment

        for month in range(1, months + 1):
            cumulative_profit += result.monthly_profit
            data.append({
                'シナリオ': result.name,
                '月': month,
                '月間コスト': result.monthly_cost,
                '月間売上': result.monthly_revenue,
                '月間利益': result.monthly_profit,
                '累積利益': cumulative_profit,
                '累積ROI': cumulative_profit / (result.monthly_cost * month + initial_investment) if (result.monthly_cost * month + initial_investment) > 0 else 0
            })

    return pd.DataFrame(data)


def calculate_financial_metrics(
    results: List[ScenarioResult],
    params: SimulationParams,
    discount_rate: float = 0.08  # 年率8%
) -> pd.DataFrame:
    """
    財務指標（NPV, IRR）を計算

    Returns:
        財務指標のDataFrame
    """
    metrics = []

    for result in results:
        # 初期投資
        initial_investment = 0
        if result.name == "C. BtoB×EEZO同時並行":
            initial_investment = params.store_setup_cost * params.eezo_partner_stores

        # 36ヶ月のキャッシュフロー
        cash_flows = [-initial_investment] + [result.monthly_profit] * 36

        npv = calculate_npv(cash_flows, discount_rate)
        irr = calculate_irr(cash_flows) * 12  # 年率換算

        metrics.append({
            'シナリオ': result.name,
            '初期投資（万円）': initial_investment / 10000,
            'NPV（万円）': npv / 10000,
            'IRR（年率）': irr,
            '投資回収期間（月）': result.break_even_months
        })

    return pd.DataFrame(metrics)


def run_multi_sensitivity_analysis(
    simulator: CostSimulator,
    params_to_analyze: List[str],
    variation_range: Tuple[float, float] = (0.7, 1.3),
    steps: int = 7
) -> Dict[str, pd.DataFrame]:
    """
    複数パラメータの感度分析を実行

    Returns:
        パラメータ別の感度分析結果
    """
    results = {}

    for param_name in params_to_analyze:
        logger.info(f"  感度分析実行中: {param_name}")
        try:
            df = simulator.sensitivity_analysis(
                param_name=param_name,
                variation_range=variation_range,
                steps=steps
            )
            results[param_name] = df
        except Exception as e:
            logger.warning(f"  {param_name}の分析でエラー: {e}")

    return results


def plot_cost_breakdown_all(
    cost_breakdown: Dict[str, Dict[str, float]],
    save_path: str
) -> None:
    """全シナリオのコスト内訳を可視化"""
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for ax, (scenario, costs) in zip(axes, cost_breakdown.items()):
        labels = list(costs.keys())
        values = list(costs.values())

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            autopct='%1.1f%%',
            colors=colors[:len(values)],
            pctdistance=0.75
        )
        ax.set_title(scenario, fontsize=11, fontweight='bold')
        ax.legend(labels, loc='upper left', bbox_to_anchor=(-0.3, 1), fontsize=8)

    plt.suptitle('コスト構造の比較', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_monthly_progression_chart(
    monthly_df: pd.DataFrame,
    save_path: str
) -> None:
    """月次収益推移グラフ"""
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # 累積利益推移
    ax = axes[0, 0]
    for i, scenario in enumerate(monthly_df['シナリオ'].unique()):
        data = monthly_df[monthly_df['シナリオ'] == scenario]
        ax.plot(data['月'], data['累積利益'] / 10000, label=scenario, color=colors[i], linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('累積利益の推移', fontsize=12, fontweight='bold')
    ax.set_xlabel('経過月数')
    ax.set_ylabel('累積利益（万円）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 月間利益比較
    ax = axes[0, 1]
    profit_data = monthly_df[monthly_df['月'] == 1][['シナリオ', '月間利益']]
    bars = ax.bar(profit_data['シナリオ'], profit_data['月間利益'] / 10000, color=colors)
    ax.set_title('月間利益の比較', fontsize=12, fontweight='bold')
    ax.set_ylabel('月間利益（万円）')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, profit_data['月間利益'] / 10000):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 累積ROI推移
    ax = axes[1, 0]
    for i, scenario in enumerate(monthly_df['シナリオ'].unique()):
        data = monthly_df[monthly_df['シナリオ'] == scenario]
        ax.plot(data['月'], data['累積ROI'] * 100, label=scenario, color=colors[i], linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('累積ROIの推移', fontsize=12, fontweight='bold')
    ax.set_xlabel('経過月数')
    ax.set_ylabel('ROI（%）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # コスト vs 売上
    ax = axes[1, 1]
    cost_revenue = monthly_df[monthly_df['月'] == 1][['シナリオ', '月間コスト', '月間売上']]
    x = np.arange(len(cost_revenue))
    width = 0.35
    bars1 = ax.bar(x - width/2, cost_revenue['月間コスト'] / 10000, width, label='コスト', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, cost_revenue['月間売上'] / 10000, width, label='売上', color='#2ecc71', alpha=0.7)
    ax.set_title('月間コスト vs 売上', fontsize=12, fontweight='bold')
    ax.set_ylabel('金額（万円）')
    ax.set_xticks(x)
    ax.set_xticklabels(cost_revenue['シナリオ'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_enhanced_report(
    results_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    cost_breakdown: Dict,
    sensitivity_results: Dict,
    output_path: str
) -> str:
    """詳細なMarkdownレポートを生成"""

    best_idx = results_df['LTV/CAC比率'].idxmax()
    best_scenario = results_df.loc[best_idx, 'シナリオ']
    best_ratio = results_df.loc[best_idx, 'LTV/CAC比率']
    target_achieved = best_ratio >= 4.0

    report = f"""# EEZO×BtoB コストシミュレーション 詳細レポート

## 1. エグゼクティブサマリー

### 推奨シナリオ
- **{best_scenario}**
- LTV/CAC比率: **{best_ratio:.2f}** （目標4.0以上: {'✓ 達成' if target_achieved else '✗ 未達成'}）

### 主要発見事項
"""

    # 各シナリオの評価
    for _, row in results_df.iterrows():
        status = "✓" if row['LTV/CAC比率'] >= 4.0 else "✗"
        report += f"- {row['シナリオ']}: LTV/CAC {row['LTV/CAC比率']:.2f} {status}\n"

    report += f"""
---

## 2. シナリオ比較

### 2.1 主要KPI一覧

{results_df.to_markdown(index=False)}

### 2.2 財務指標

{financial_df.to_markdown(index=False)}

---

## 3. コスト構造分析

"""

    for scenario, costs in cost_breakdown.items():
        total = sum(costs.values())
        report += f"### {scenario}\n\n"
        report += "| コスト項目 | 金額（円/月） | 構成比 |\n"
        report += "|-----------|--------------|--------|\n"
        for item, value in costs.items():
            pct = (value / total * 100) if total > 0 else 0
            report += f"| {item} | {value:,.0f} | {pct:.1f}% |\n"
        report += f"| **合計** | **{total:,.0f}** | **100%** |\n\n"

    report += """
---

## 4. 感度分析結果

"""

    param_labels = {
        'eezo_cvr': 'EC CVR（QR→購入転換率）',
        'eezo_customer_price': '顧客単価',
        'eezo_repeat_rate': 'リピート率',
        'btob_conversion_rate': 'BtoB成約率',
        'synergy_ltv_multiplier': 'シナジーLTV倍率'
    }

    for param_name, df in sensitivity_results.items():
        label = param_labels.get(param_name, param_name)
        report += f"### {label}\n\n"

        # シナリオC（同時並行）のデータを抽出
        combined_data = df[df['scenario'] == 'C. BtoB×EEZO同時並行']
        if not combined_data.empty:
            min_mult = combined_data['multiplier'].min()
            max_mult = combined_data['multiplier'].max()
            min_ratio = combined_data[combined_data['multiplier'] == min_mult]['ltv_cac_ratio'].values[0]
            max_ratio = combined_data[combined_data['multiplier'] == max_mult]['ltv_cac_ratio'].values[0]

            report += f"- 変動範囲: {min_mult*100:.0f}%～{max_mult*100:.0f}%\n"
            report += f"- LTV/CAC比率の変化: {min_ratio:.2f}～{max_ratio:.2f}\n\n"

    target_status = '目標の4.0以上を達成しています。' if target_achieved else '目標の4.0には届いていませんが、改善余地があります。'
    analysis_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    report += f"""
---

## 5. リスクと推奨事項

### 5.1 主要リスク
1. **CVR未達リスク**: QR→購入転換率が8%を下回ると収益性が大幅に低下
2. **シナジー効果の不確実性**: 同時並行モデルのシナジー効果は仮説段階
3. **店舗パートナーシップの継続性**: 店舗との関係維持コストが増加する可能性

### 5.2 推奨アクション
1. **パイロット検証**: 3-5店舗での小規模実証を推奨
2. **KPIモニタリング**: CVR、リピート率の週次トラッキング体制構築
3. **段階的展開**: 実績に基づく店舗数拡大計画の策定

---

## 6. 結論

{best_scenario}が最も高いLTV/CAC比率（{best_ratio:.2f}）を示しており、
{target_status}

同時並行モデルは、BtoB営業とEC運営のシナジー効果により、
単独モデルと比較して高い収益性を実現できる可能性があります。

ただし、シナジー効果の実現には店舗との強固なパートナーシップ構築と、
効果的なクロスセル戦略の実行が不可欠です。

---

*Generated by EEZO-BtoB Cost Simulator*
*Analysis Date: {analysis_date}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


def main():
    """メイン実行関数"""
    logger.info("=" * 60)
    logger.info("EEZO×BtoB 同時並行モデル シミュレーション開始")
    logger.info("=" * 60)

    # 出力ディレクトリ設定
    output_dir = project_root / 'outputs'
    processed_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    # Step 1: パラメータ設定
    logger.info("\n[Step 1] パラメータ読み込み")
    params_path = project_root / 'data' / 'raw' / 'parameters.csv'

    try:
        params = load_params_from_csv(str(params_path))
        logger.info("  CSVからパラメータを読み込みました")
    except Exception as e:
        logger.warning(f"  CSV読み込みエラー: {e}")
        logger.info("  デフォルトパラメータを使用します")
        params = SimulationParams()

    # パラメータ確認
    logger.info(f"  BtoB月間訪問数: {params.btob_monthly_visits}件")
    logger.info(f"  BtoB成約率: {params.btob_conversion_rate:.1%}")
    logger.info(f"  EEZO CVR: {params.eezo_cvr:.1%}")
    logger.info(f"  EEZO顧客単価: {format_currency(params.eezo_customer_price, '円')}")
    logger.info(f"  シナジーLTV倍率: {params.synergy_ltv_multiplier:.2f}x")

    # Step 2: シミュレーション実行
    logger.info("\n[Step 2] 3シナリオ シミュレーション実行")
    simulator = CostSimulator(params)
    results = simulator.run_all_scenarios()

    for r in results:
        logger.info(f"  {r.name}:")
        logger.info(f"    月間コスト: {format_currency(r.monthly_cost, '万円', 1)}")
        logger.info(f"    月間売上: {format_currency(r.monthly_revenue, '万円', 1)}")
        logger.info(f"    LTV/CAC比率: {r.ltv_cac_ratio:.2f}")
        logger.info(f"    3年ROI: {r.roi_3year:.1%}")

    # Step 3: DataFrame変換・保存
    logger.info("\n[Step 3] 結果の保存")
    results_df = simulator.results_to_dataframe(results)
    csv_path = processed_dir / 'simulation_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"  CSV保存: {csv_path}")

    # Step 4: コスト構造分析
    logger.info("\n[Step 4] コスト構造分析")
    cost_breakdown = analyze_cost_breakdown(params)
    for scenario, costs in cost_breakdown.items():
        total = sum(costs.values())
        logger.info(f"  {scenario}: 月間総コスト {format_currency(total, '万円', 1)}")

    # Step 5: 財務指標計算
    logger.info("\n[Step 5] 財務指標計算（NPV, IRR）")
    financial_df = calculate_financial_metrics(results, params)
    for _, row in financial_df.iterrows():
        logger.info(f"  {row['シナリオ']}:")
        logger.info(f"    NPV: {row['NPV（万円）']:.1f}万円")
        logger.info(f"    IRR: {row['IRR（年率）']:.1%}")

    # Step 6: 月次推移分析
    logger.info("\n[Step 6] 月次収益推移分析")
    monthly_df = analyze_monthly_progression(results, params, months=36)
    monthly_csv = processed_dir / 'monthly_progression.csv'
    monthly_df.to_csv(monthly_csv, index=False, encoding='utf-8-sig')
    logger.info(f"  月次データ保存: {monthly_csv}")

    # Step 7: 感度分析
    logger.info("\n[Step 7] 感度分析実行")
    sensitivity_params = [
        'eezo_cvr',
        'eezo_customer_price',
        'eezo_repeat_rate',
        'btob_conversion_rate',
        'synergy_ltv_multiplier'
    ]
    sensitivity_results = run_multi_sensitivity_analysis(
        simulator, sensitivity_params, variation_range=(0.7, 1.3), steps=7
    )
    logger.info(f"  {len(sensitivity_results)}パラメータの感度分析完了")

    # Step 8: 可視化
    logger.info("\n[Step 8] 可視化")

    # シナリオ比較
    plot_scenario_comparison(results_df, save_path=str(output_dir / 'scenario_comparison.png'))
    logger.info("  シナリオ比較グラフ保存")

    # LTV/CAC比較
    plot_ltv_cac_comparison(results_df, target_ratio=4.0, save_path=str(output_dir / 'ltv_cac_comparison.png'))
    logger.info("  LTV/CAC比較グラフ保存")

    # 累積利益推移
    plot_cumulative_profit(results, months=36, save_path=str(output_dir / 'cumulative_profit.png'))
    logger.info("  累積利益推移グラフ保存")

    # コスト内訳
    plot_cost_breakdown_all(cost_breakdown, save_path=str(output_dir / 'cost_breakdown.png'))
    logger.info("  コスト内訳グラフ保存")

    # 月次推移
    plot_monthly_progression_chart(monthly_df, save_path=str(output_dir / 'monthly_progression.png'))
    logger.info("  月次推移グラフ保存")

    # 感度分析ヒートマップ
    for param_name, df in sensitivity_results.items():
        plot_sensitivity_heatmap(df, metric='ltv_cac_ratio',
                                save_path=str(output_dir / f'sensitivity_{param_name}.png'))
    logger.info("  感度分析ヒートマップ保存")

    # Step 9: レポート生成
    logger.info("\n[Step 9] 詳細レポート生成")
    report = generate_enhanced_report(
        results_df, financial_df, cost_breakdown, sensitivity_results,
        output_path=str(output_dir / 'simulation_report.md')
    )
    logger.info(f"  レポート保存: {output_dir / 'simulation_report.md'}")

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("シミュレーション完了")
    logger.info("=" * 60)

    print("\n" + "=" * 70)
    print("【シミュレーション結果サマリー】")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\n" + "-" * 70)
    print("【財務指標】")
    print("-" * 70)
    print(financial_df.to_string(index=False))
    print("\n" + "-" * 70)

    # 推奨シナリオ
    best_idx = results_df['LTV/CAC比率'].idxmax()
    best_scenario = results_df.loc[best_idx, 'シナリオ']
    best_ratio = results_df.loc[best_idx, 'LTV/CAC比率']

    print(f"\n【推奨シナリオ】")
    print(f"  {best_scenario}")
    print(f"  LTV/CAC比率: {best_ratio:.2f}")
    print(f"  目標（4.0以上）達成: {'✓' if best_ratio >= 4.0 else '✗'}")
    print("=" * 70)

    return results_df, financial_df, results


if __name__ == "__main__":
    results_df, financial_df, results = main()
