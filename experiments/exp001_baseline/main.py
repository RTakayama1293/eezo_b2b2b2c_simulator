"""
exp001_baseline: 3シナリオの基準値シミュレーション

目的:
- BtoB単独、EEZO単独、同時並行モデルの経済性比較
- LTV/CAC比率の目標達成可否確認
- 感度分析による主要パラメータの影響度評価
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.simulator import SimulationParams, CostSimulator
from src.visualizer import (
    plot_scenario_comparison,
    plot_ltv_cac_comparison,
    plot_cumulative_profit,
    plot_sensitivity_heatmap,
    generate_summary_report
)
from src.utils import setup_logger, format_currency

# ロガー設定
logger = setup_logger('exp001')


def main():
    """メイン実行関数"""
    logger.info("=== exp001_baseline 開始 ===")
    
    # 出力ディレクトリ確認
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Step 1: パラメータ設定
    logger.info("Step 1: パラメータ設定")
    params = SimulationParams()
    
    # パラメータ確認
    logger.info(f"  BtoB月間訪問数: {params.btob_monthly_visits}件")
    logger.info(f"  BtoB成約率: {params.btob_conversion_rate:.1%}")
    logger.info(f"  EEZO CVR: {params.eezo_cvr:.1%}")
    logger.info(f"  EEZO顧客単価: {format_currency(params.eezo_customer_price, '円')}")
    
    # Step 2: シミュレーション実行
    logger.info("Step 2: シミュレーション実行")
    simulator = CostSimulator(params)
    results = simulator.run_all_scenarios()
    
    # 結果表示
    for r in results:
        logger.info(f"  {r.name}:")
        logger.info(f"    LTV/CAC比率: {r.ltv_cac_ratio:.2f}")
        logger.info(f"    月間利益: {format_currency(r.monthly_profit, '万円', 1)}")
        logger.info(f"    3年ROI: {r.roi_3year:.1%}")
    
    # Step 3: DataFrame変換
    logger.info("Step 3: 結果をDataFrame変換")
    df = simulator.results_to_dataframe(results)
    
    # CSV保存
    csv_path = processed_dir / 'results_exp001.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"  結果保存: {csv_path}")
    
    # Step 4: 可視化
    logger.info("Step 4: 可視化")
    
    # シナリオ比較
    plot_scenario_comparison(
        df,
        save_path=str(output_dir / 'scenario_comparison.png')
    )
    logger.info("  シナリオ比較グラフ保存")
    
    # LTV/CAC比較
    plot_ltv_cac_comparison(
        df,
        target_ratio=4.0,
        save_path=str(output_dir / 'ltv_cac_comparison.png')
    )
    logger.info("  LTV/CAC比較グラフ保存")
    
    # 累積利益推移
    plot_cumulative_profit(
        results,
        months=36,
        save_path=str(output_dir / 'cumulative_profit.png')
    )
    logger.info("  累積利益推移グラフ保存")
    
    # Step 5: 感度分析
    logger.info("Step 5: 感度分析")
    
    sensitivity_params = ['eezo_cvr', 'eezo_customer_price', 'eezo_repeat_rate']
    
    for param in sensitivity_params:
        logger.info(f"  分析中: {param}")
        sensitivity_df = simulator.sensitivity_analysis(
            param_name=param,
            variation_range=(0.7, 1.3),
            steps=7
        )
        
        # ヒートマップ保存
        plot_sensitivity_heatmap(
            sensitivity_df,
            metric='ltv_cac_ratio',
            save_path=str(output_dir / f'sensitivity_{param}.png')
        )
    
    logger.info("  感度分析完了")
    
    # Step 6: レポート生成
    logger.info("Step 6: レポート生成")
    report = generate_summary_report(
        df,
        output_path=str(output_dir / 'simulation_report.md')
    )
    logger.info(f"  レポート保存: {output_dir / 'simulation_report.md'}")
    
    # サマリー表示
    logger.info("=== 実行完了 ===")
    print("\n" + "="*60)
    print("シミュレーション結果サマリー")
    print("="*60)
    print(df.to_string(index=False))
    print("\n" + "="*60)
    
    # 推奨シナリオ
    best_idx = df['LTV/CAC比率'].idxmax()
    best_scenario = df.loc[best_idx, 'シナリオ']
    best_ratio = df.loc[best_idx, 'LTV/CAC比率']
    
    print(f"\n推奨シナリオ: {best_scenario}")
    print(f"LTV/CAC比率: {best_ratio:.2f}")
    
    target_achieved = best_ratio >= 4.0
    print(f"目標（4.0以上）達成: {'✓' if target_achieved else '×'}")
    
    return df, results


if __name__ == "__main__":
    main()
