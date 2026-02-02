"""
EEZO×BtoB 同時並行モデル コストシミュレーター

3つのシナリオを比較分析：
- A: BtoB営業単独モデル
- B: EEZO（EC）単独モデル  
- C: BtoB×EEZO同時並行モデル

主要KPI:
- CAC (Customer Acquisition Cost)
- LTV (Lifetime Value)
- ROI (Return on Investment)
- BEP (Break Even Point)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class SimulationParams:
    """シミュレーションパラメータを保持するデータクラス"""
    
    # BtoB営業パラメータ
    btob_monthly_salary: float = 400_000  # 円/月
    btob_monthly_visits: int = 20  # 件/月
    btob_visit_cost: float = 10_000  # 円/件（交通費+サンプル+接待）
    btob_conversion_rate: float = 0.10  # 成約率
    btob_avg_annual_deal: float = 1_200_000  # 円/年
    btob_retention_rate: float = 0.80  # 継続率
    btob_customer_lifetime: int = 36  # 月
    btob_deal_cost: float = 30_000  # 円/件（契約事務費）
    
    # EEZO（EC）パラメータ
    eezo_partner_stores: int = 10  # 店舗数
    eezo_qr_scans_per_store: int = 100  # スキャン数/店舗/月
    eezo_cvr: float = 0.08  # QR→購入CVR
    eezo_customer_price: float = 8_000  # 円
    eezo_repeat_rate: float = 0.25  # リピート率
    eezo_purchase_frequency: float = 2.5  # 回/年
    eezo_customer_lifetime: int = 24  # 月
    
    # EEZOコストパラメータ
    eezo_platform_fee: float = 50_000  # 円/月
    eezo_ad_budget: float = 300_000  # 円/月
    eezo_shipping_cost: float = 1_200  # 円/件
    eezo_packing_cost: float = 300  # 円/件
    eezo_commission_rate: float = 0.15  # 店舗インセンティブ
    
    # 同時並行モデルパラメータ
    store_setup_cost: float = 50_000  # 円/店舗
    store_monthly_incentive: float = 30_000  # 円/月/店舗
    coordination_cost: float = 100_000  # 円/月
    
    # シナジー効果
    synergy_btob_cvr_boost: float = 1.2  # BtoB成約率向上
    synergy_eezo_cvr_boost: float = 1.3  # EC CVR向上
    synergy_ltv_multiplier: float = 1.25  # LTV向上
    synergy_cac_efficiency: float = 0.85  # コスト効率化


def load_params_from_csv(filepath: str) -> SimulationParams:
    """
    CSVファイルからパラメータを読み込む
    
    Args:
        filepath: パラメータCSVファイルのパス
        
    Returns:
        SimulationParams: パラメータオブジェクト
    """
    df = pd.read_csv(filepath, comment='#')
    params = SimulationParams()
    
    param_mapping = {
        ('btob', 'monthly_salary'): 'btob_monthly_salary',
        ('btob', 'monthly_visits'): 'btob_monthly_visits',
        ('btob', 'visit_cost'): 'btob_visit_cost',
        ('btob', 'conversion_rate'): 'btob_conversion_rate',
        ('btob', 'avg_annual_deal'): 'btob_avg_annual_deal',
        ('btob', 'retention_rate'): 'btob_retention_rate',
        ('btob', 'customer_lifetime'): 'btob_customer_lifetime',
        ('eezo', 'partner_stores'): 'eezo_partner_stores',
        ('eezo', 'cvr_qr_to_purchase'): 'eezo_cvr',
        ('eezo', 'customer_unit_price'): 'eezo_customer_price',
        ('eezo', 'repeat_rate_3m'): 'eezo_repeat_rate',
        ('eezo', 'avg_purchase_frequency'): 'eezo_purchase_frequency',
        ('eezo', 'customer_lifetime'): 'eezo_customer_lifetime',
        ('cost_eezo', 'platform_fee'): 'eezo_platform_fee',
        ('cost_eezo', 'ad_budget'): 'eezo_ad_budget',
        ('cost_eezo', 'shipping_per_order'): 'eezo_shipping_cost',
        ('cost_eezo', 'packing_per_order'): 'eezo_packing_cost',
        ('cost_eezo', 'commission_rate'): 'eezo_commission_rate',
        ('cost_combined', 'store_setup_cost'): 'store_setup_cost',
        ('cost_combined', 'monthly_incentive_per_store'): 'store_monthly_incentive',
        ('cost_combined', 'coordination_cost'): 'coordination_cost',
        ('synergy', 'btob_cvr_boost'): 'synergy_btob_cvr_boost',
        ('synergy', 'eezo_cvr_boost'): 'synergy_eezo_cvr_boost',
        ('synergy', 'ltv_multiplier'): 'synergy_ltv_multiplier',
        ('synergy', 'cac_efficiency'): 'synergy_cac_efficiency',
    }
    
    for (cat, name), attr in param_mapping.items():
        row = df[(df['category'] == cat) & (df['param_name'] == name)]
        if not row.empty:
            setattr(params, attr, row['base_value'].values[0])
    
    return params


@dataclass
class ScenarioResult:
    """シナリオ分析結果を保持するデータクラス"""
    name: str
    monthly_cost: float  # 月間コスト
    monthly_revenue: float  # 月間売上
    monthly_new_customers: float  # 月間新規顧客数
    cac: float  # 顧客獲得コスト
    ltv: float  # 顧客生涯価値
    ltv_cac_ratio: float  # LTV/CAC比率
    monthly_profit: float  # 月間利益
    annual_profit: float  # 年間利益
    break_even_months: Optional[int]  # 損益分岐月数
    roi_3year: float  # 3年ROI


class CostSimulator:
    """コストシミュレーションのメインクラス"""
    
    def __init__(self, params: SimulationParams):
        """
        Args:
            params: シミュレーションパラメータ
        """
        self.params = params
    
    def simulate_btob_only(self) -> ScenarioResult:
        """
        シナリオA: BtoB営業単独モデルのシミュレーション
        
        計算ロジック:
        - 月間コスト = 固定費 + 訪問変動費 + 成約変動費
        - 月間新規顧客 = 訪問数 × 成約率
        - 月間売上 = 既存顧客売上 + 新規顧客売上（初月）
        - LTV = 年間取引額 × 継続期間（年）× 継続率
        """
        p = self.params
        
        # 月間コスト計算
        fixed_cost = p.btob_monthly_salary + 25_000  # 人件費+諸経費
        visit_cost = p.btob_monthly_visits * p.btob_visit_cost
        new_deals = p.btob_monthly_visits * p.btob_conversion_rate
        deal_cost = new_deals * p.btob_deal_cost
        monthly_cost = fixed_cost + visit_cost + deal_cost
        
        # 月間新規顧客
        monthly_new_customers = new_deals
        
        # CAC計算
        cac = monthly_cost / monthly_new_customers if monthly_new_customers > 0 else float('inf')
        
        # LTV計算（BtoB）
        # LTV = 年間取引額 × Σ(継続率^n) for n=0 to lifetime_years-1
        lifetime_years = p.btob_customer_lifetime / 12
        retention_sum = sum(p.btob_retention_rate ** n for n in range(int(lifetime_years)))
        ltv = p.btob_avg_annual_deal * retention_sum
        
        # 月間売上（定常状態想定：累積顧客×月間売上）
        # 簡略化：新規顧客が毎月一定で増加する前提での定常状態
        monthly_revenue_per_customer = p.btob_avg_annual_deal / 12
        # 定常状態の顧客数 = 月間新規 × 平均継続月数 × 継続率の影響
        steady_state_customers = monthly_new_customers * p.btob_customer_lifetime * 0.7  # 簡略化
        monthly_revenue = steady_state_customers * monthly_revenue_per_customer
        
        # 利益計算
        monthly_profit = monthly_revenue - monthly_cost
        annual_profit = monthly_profit * 12
        
        # LTV/CAC比率
        ltv_cac_ratio = ltv / cac if cac > 0 else 0
        
        # 損益分岐点（累積利益がプラスになる月）
        break_even_months = self._calculate_break_even(monthly_cost, monthly_revenue)
        
        # 3年ROI
        total_cost_3y = monthly_cost * 36
        total_revenue_3y = monthly_revenue * 36
        roi_3year = (total_revenue_3y - total_cost_3y) / total_cost_3y if total_cost_3y > 0 else 0
        
        return ScenarioResult(
            name="A. BtoB営業単独",
            monthly_cost=monthly_cost,
            monthly_revenue=monthly_revenue,
            monthly_new_customers=monthly_new_customers,
            cac=cac,
            ltv=ltv,
            ltv_cac_ratio=ltv_cac_ratio,
            monthly_profit=monthly_profit,
            annual_profit=annual_profit,
            break_even_months=break_even_months,
            roi_3year=roi_3year
        )
    
    def simulate_eezo_only(self) -> ScenarioResult:
        """
        シナリオB: EEZO（EC）単独モデルのシミュレーション
        
        計算ロジック:
        - 月間コスト = プラットフォーム費 + 広告費 + 変動費（配送・梱包・インセンティブ）
        - 月間新規顧客 = QRスキャン数 × CVR
        - LTV = 顧客単価 × 購入頻度 × 継続期間
        """
        p = self.params
        
        # 月間QRスキャン数
        total_qr_scans = p.eezo_partner_stores * p.eezo_qr_scans_per_store
        
        # 月間新規顧客（EC）
        monthly_new_customers = total_qr_scans * p.eezo_cvr
        
        # 月間注文数（新規 + リピート）
        # リピーターからの注文も含む
        monthly_orders = monthly_new_customers * (1 + p.eezo_repeat_rate * (p.eezo_purchase_frequency - 1) / 12)
        
        # 月間売上
        monthly_revenue = monthly_orders * p.eezo_customer_price
        
        # 月間コスト
        fixed_cost = p.eezo_platform_fee + p.eezo_ad_budget
        variable_cost = monthly_orders * (p.eezo_shipping_cost + p.eezo_packing_cost)
        commission = monthly_revenue * p.eezo_commission_rate
        monthly_cost = fixed_cost + variable_cost + commission
        
        # CAC計算（広告費ベース）
        cac = (p.eezo_ad_budget + commission) / monthly_new_customers if monthly_new_customers > 0 else float('inf')
        
        # LTV計算（EC）
        annual_revenue_per_customer = p.eezo_customer_price * p.eezo_purchase_frequency
        lifetime_years = p.eezo_customer_lifetime / 12
        # 減衰を考慮したLTV
        ltv = annual_revenue_per_customer * lifetime_years * (1 + p.eezo_repeat_rate) / 2
        
        # LTV/CAC比率
        ltv_cac_ratio = ltv / cac if cac > 0 else 0
        
        # 利益計算
        monthly_profit = monthly_revenue - monthly_cost
        annual_profit = monthly_profit * 12
        
        # 損益分岐点
        break_even_months = self._calculate_break_even(monthly_cost, monthly_revenue)
        
        # 3年ROI
        total_cost_3y = monthly_cost * 36
        total_revenue_3y = monthly_revenue * 36
        roi_3year = (total_revenue_3y - total_cost_3y) / total_cost_3y if total_cost_3y > 0 else 0
        
        return ScenarioResult(
            name="B. EEZO単独",
            monthly_cost=monthly_cost,
            monthly_revenue=monthly_revenue,
            monthly_new_customers=monthly_new_customers,
            cac=cac,
            ltv=ltv,
            ltv_cac_ratio=ltv_cac_ratio,
            monthly_profit=monthly_profit,
            annual_profit=annual_profit,
            break_even_months=break_even_months,
            roi_3year=roi_3year
        )
    
    def simulate_combined(self) -> ScenarioResult:
        """
        シナリオC: BtoB×EEZO同時並行モデルのシミュレーション
        
        計算ロジック:
        - シナジー効果を適用したBtoB + EEZO
        - 追加コスト: 店舗連携費用、コーディネーション費用
        - シナジー: CVR向上、LTV向上、コスト効率化
        """
        p = self.params
        
        # --- BtoB部分（シナジー適用）---
        btob_cvr = p.btob_conversion_rate * p.synergy_btob_cvr_boost
        btob_new_deals = p.btob_monthly_visits * btob_cvr
        
        btob_fixed_cost = p.btob_monthly_salary + 25_000
        btob_visit_cost = p.btob_monthly_visits * p.btob_visit_cost
        btob_deal_cost = btob_new_deals * p.btob_deal_cost
        btob_monthly_cost = (btob_fixed_cost + btob_visit_cost + btob_deal_cost) * p.synergy_cac_efficiency
        
        btob_monthly_revenue_per_customer = p.btob_avg_annual_deal / 12
        btob_steady_customers = btob_new_deals * p.btob_customer_lifetime * 0.7
        btob_monthly_revenue = btob_steady_customers * btob_monthly_revenue_per_customer
        
        # --- EEZO部分（シナジー適用）---
        eezo_cvr = p.eezo_cvr * p.synergy_eezo_cvr_boost
        total_qr_scans = p.eezo_partner_stores * p.eezo_qr_scans_per_store
        eezo_new_customers = total_qr_scans * eezo_cvr
        
        eezo_monthly_orders = eezo_new_customers * (1 + p.eezo_repeat_rate * (p.eezo_purchase_frequency - 1) / 12)
        eezo_monthly_revenue = eezo_monthly_orders * p.eezo_customer_price
        
        eezo_fixed_cost = (p.eezo_platform_fee + p.eezo_ad_budget * 0.5) * p.synergy_cac_efficiency  # 広告費削減
        eezo_variable_cost = eezo_monthly_orders * (p.eezo_shipping_cost + p.eezo_packing_cost)
        eezo_commission = eezo_monthly_revenue * p.eezo_commission_rate
        eezo_monthly_cost = eezo_fixed_cost + eezo_variable_cost + eezo_commission
        
        # --- 同時並行モデル追加コスト ---
        combined_additional_cost = (
            p.store_monthly_incentive * p.eezo_partner_stores +
            p.coordination_cost
        )
        
        # --- 合計 ---
        monthly_cost = btob_monthly_cost + eezo_monthly_cost + combined_additional_cost
        monthly_revenue = btob_monthly_revenue + eezo_monthly_revenue
        monthly_new_customers = btob_new_deals + eezo_new_customers
        
        # CAC
        cac = monthly_cost / monthly_new_customers if monthly_new_customers > 0 else float('inf')
        
        # LTV（シナジー効果で向上）
        btob_ltv = p.btob_avg_annual_deal * (p.btob_customer_lifetime / 12) * 0.7
        eezo_ltv = p.eezo_customer_price * p.eezo_purchase_frequency * (p.eezo_customer_lifetime / 12)
        # 加重平均LTV × シナジー
        ltv = ((btob_ltv * btob_new_deals + eezo_ltv * eezo_new_customers) / 
               monthly_new_customers * p.synergy_ltv_multiplier) if monthly_new_customers > 0 else 0
        
        # LTV/CAC比率
        ltv_cac_ratio = ltv / cac if cac > 0 else 0
        
        # 利益計算
        monthly_profit = monthly_revenue - monthly_cost
        annual_profit = monthly_profit * 12
        
        # 損益分岐点
        initial_investment = p.store_setup_cost * p.eezo_partner_stores
        break_even_months = self._calculate_break_even(
            monthly_cost, monthly_revenue, initial_investment
        )
        
        # 3年ROI
        total_cost_3y = monthly_cost * 36 + initial_investment
        total_revenue_3y = monthly_revenue * 36
        roi_3year = (total_revenue_3y - total_cost_3y) / total_cost_3y if total_cost_3y > 0 else 0
        
        return ScenarioResult(
            name="C. BtoB×EEZO同時並行",
            monthly_cost=monthly_cost,
            monthly_revenue=monthly_revenue,
            monthly_new_customers=monthly_new_customers,
            cac=cac,
            ltv=ltv,
            ltv_cac_ratio=ltv_cac_ratio,
            monthly_profit=monthly_profit,
            annual_profit=annual_profit,
            break_even_months=break_even_months,
            roi_3year=roi_3year
        )
    
    def _calculate_break_even(
        self, 
        monthly_cost: float, 
        monthly_revenue: float,
        initial_investment: float = 0
    ) -> Optional[int]:
        """
        損益分岐月数を計算
        
        Args:
            monthly_cost: 月間コスト
            monthly_revenue: 月間売上
            initial_investment: 初期投資額
            
        Returns:
            損益分岐月数（達成不可の場合はNone）
        """
        monthly_profit = monthly_revenue - monthly_cost
        if monthly_profit <= 0:
            return None
        
        # 累積利益が初期投資を回収する月数
        if initial_investment > 0:
            return int(np.ceil(initial_investment / monthly_profit))
        else:
            return 1 if monthly_profit > 0 else None
    
    def run_all_scenarios(self) -> List[ScenarioResult]:
        """全シナリオを実行して結果を返す"""
        return [
            self.simulate_btob_only(),
            self.simulate_eezo_only(),
            self.simulate_combined()
        ]
    
    def sensitivity_analysis(
        self, 
        param_name: str, 
        variation_range: Tuple[float, float] = (0.7, 1.3),
        steps: int = 11
    ) -> pd.DataFrame:
        """
        感度分析を実行
        
        Args:
            param_name: 分析対象のパラメータ名
            variation_range: 変動範囲（基準値に対する倍率）
            steps: 分析ステップ数
            
        Returns:
            感度分析結果のDataFrame
        """
        results = []
        base_value = getattr(self.params, param_name)
        multipliers = np.linspace(variation_range[0], variation_range[1], steps)
        
        for mult in multipliers:
            # パラメータを一時的に変更
            test_value = base_value * mult
            setattr(self.params, param_name, test_value)
            
            # シミュレーション実行
            scenarios = self.run_all_scenarios()
            
            for scenario in scenarios:
                results.append({
                    'param_name': param_name,
                    'multiplier': mult,
                    'param_value': test_value,
                    'scenario': scenario.name,
                    'ltv_cac_ratio': scenario.ltv_cac_ratio,
                    'roi_3year': scenario.roi_3year,
                    'monthly_profit': scenario.monthly_profit
                })
        
        # パラメータを元に戻す
        setattr(self.params, param_name, base_value)
        
        return pd.DataFrame(results)
    
    def results_to_dataframe(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        data = []
        for r in results:
            data.append({
                'シナリオ': r.name,
                '月間コスト（万円）': r.monthly_cost / 10000,
                '月間売上（万円）': r.monthly_revenue / 10000,
                '月間新規顧客数': r.monthly_new_customers,
                'CAC（円）': r.cac,
                'LTV（円）': r.ltv,
                'LTV/CAC比率': r.ltv_cac_ratio,
                '月間利益（万円）': r.monthly_profit / 10000,
                '年間利益（万円）': r.annual_profit / 10000,
                '損益分岐月数': r.break_even_months,
                '3年ROI': r.roi_3year
            })
        return pd.DataFrame(data)


if __name__ == "__main__":
    # テスト実行
    params = SimulationParams()
    simulator = CostSimulator(params)
    results = simulator.run_all_scenarios()
    
    df = simulator.results_to_dataframe(results)
    print(df.to_string())
