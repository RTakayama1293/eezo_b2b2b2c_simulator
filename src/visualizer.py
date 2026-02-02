"""
シミュレーション結果の可視化モジュール

提供する可視化:
- シナリオ比較バーチャート
- LTV/CAC比率の比較
- 感度分析ヒートマップ
- 損益推移グラフ
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def set_style():
    """グラフスタイルを設定"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_scenario_comparison(
    df: pd.DataFrame,
    metrics: List[str] = ['月間利益（万円）', 'LTV/CAC比率', '3年ROI'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    シナリオ比較のバーチャートを作成
    
    Args:
        df: results_to_dataframe()の出力
        metrics: 比較するメトリクス
        save_path: 保存先パス（Noneの場合は保存しない）
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
    
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df['シナリオ'], df[metric], color=colors)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, df[metric]):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ltv_cac_comparison(
    df: pd.DataFrame,
    target_ratio: float = 4.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    LTV/CAC比率の比較グラフ（目標ラインつき）
    
    Args:
        df: results_to_dataframe()の出力
        target_ratio: 目標LTV/CAC比率
        save_path: 保存先パス
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(df['シナリオ'], df['LTV/CAC比率'], color=colors)
    
    # 目標ライン
    ax.axhline(y=target_ratio, color='#e67e22', linestyle='--', linewidth=2, label=f'目標: {target_ratio}:1')
    
    ax.set_title('LTV/CAC比率の比較', fontsize=16, fontweight='bold')
    ax.set_ylabel('LTV/CAC比率')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # 値をバーの上に表示
    for bar, val in zip(bars, df['LTV/CAC比率']):
        height = bar.get_height()
        color = '#27ae60' if val >= target_ratio else '#c0392b'
        ax.annotate(f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    metric: str = 'ltv_cac_ratio',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    感度分析のヒートマップを作成
    
    Args:
        sensitivity_df: sensitivity_analysis()の出力
        metric: 可視化するメトリクス
        save_path: 保存先パス
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    # ピボットテーブル作成
    pivot = sensitivity_df.pivot_table(
        index='multiplier',
        columns='scenario',
        values=metric
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # ヒートマップ
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=4.0 if metric == 'ltv_cac_ratio' else 0,
        ax=ax
    )
    
    metric_names = {
        'ltv_cac_ratio': 'LTV/CAC比率',
        'roi_3year': '3年ROI',
        'monthly_profit': '月間利益（円）'
    }
    
    ax.set_title(f'感度分析: {metric_names.get(metric, metric)}', fontsize=16, fontweight='bold')
    ax.set_ylabel('パラメータ変動率')
    ax.set_xlabel('シナリオ')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cumulative_profit(
    results: List,
    months: int = 36,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    累積利益の推移グラフを作成
    
    Args:
        results: シミュレーション結果のリスト
        months: シミュレーション期間（月）
        save_path: 保存先パス
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for result, color in zip(results, colors):
        monthly_profits = [result.monthly_profit] * months
        cumulative = np.cumsum(monthly_profits)
        ax.plot(range(1, months + 1), cumulative / 10000, 
                label=result.name, color=color, linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_title('累積利益の推移（3年間）', fontsize=16, fontweight='bold')
    ax.set_xlabel('経過月数')
    ax.set_ylabel('累積利益（万円）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cost_breakdown(
    scenario_name: str,
    cost_items: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    コスト内訳の円グラフを作成
    
    Args:
        scenario_name: シナリオ名
        cost_items: コスト項目の辞書 {項目名: 金額}
        save_path: 保存先パス
        
    Returns:
        matplotlib Figure
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(cost_items.keys())
    values = list(cost_items.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        pctdistance=0.75,
        startangle=90
    )
    
    ax.set_title(f'{scenario_name}\nコスト内訳', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_summary_report(
    df: pd.DataFrame,
    output_path: str = 'outputs/simulation_report.md'
) -> str:
    """
    シミュレーション結果のMarkdownレポートを生成
    
    Args:
        df: results_to_dataframe()の出力
        output_path: 出力ファイルパス
        
    Returns:
        生成したレポートの文字列
    """
    best_scenario = df.loc[df['LTV/CAC比率'].idxmax(), 'シナリオ']
    best_ratio = df['LTV/CAC比率'].max()
    
    report = f"""# EEZO×BtoB コストシミュレーション結果

## エグゼクティブサマリー

**推奨シナリオ**: {best_scenario}  
**LTV/CAC比率**: {best_ratio:.2f}（目標4.0以上）

## シナリオ比較

{df.to_markdown(index=False)}

## 主要な発見

1. **LTV/CAC比率**: 
   - 目標の4.0以上を達成しているシナリオ: {', '.join(df[df['LTV/CAC比率'] >= 4.0]['シナリオ'].tolist()) or 'なし'}

2. **投資回収期間**:
"""
    
    for _, row in df.iterrows():
        bep = row['損益分岐月数']
        bep_str = f"{bep}ヶ月" if bep else "達成不可"
        report += f"   - {row['シナリオ']}: {bep_str}\n"
    
    report += f"""
3. **3年ROI**:
"""
    
    for _, row in df.iterrows():
        report += f"   - {row['シナリオ']}: {row['3年ROI']:.1%}\n"
    
    report += """
## 次のアクション

1. パラメータの精緻化（実績データでの検証）
2. 感度分析による リスク評価
3. パイロット実施計画の策定

---
*Generated by EEZO-BtoB Cost Simulator*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    # テスト用データ
    test_data = {
        'シナリオ': ['A. BtoB営業単独', 'B. EEZO単独', 'C. BtoB×EEZO同時並行'],
        '月間コスト（万円）': [45, 40, 75],
        '月間売上（万円）': [80, 60, 150],
        '月間新規顧客数': [2, 80, 100],
        'CAC（円）': [225000, 5000, 7500],
        'LTV（円）': [2100000, 50000, 250000],
        'LTV/CAC比率': [9.3, 10.0, 33.3],
        '月間利益（万円）': [35, 20, 75],
        '年間利益（万円）': [420, 240, 900],
        '損益分岐月数': [1, 1, 1],
        '3年ROI': [1.5, 1.0, 2.0]
    }
    df = pd.DataFrame(test_data)
    
    # グラフ生成テスト
    plot_scenario_comparison(df)
    plt.show()
