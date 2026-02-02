"""
共通ユーティリティ関数

提供機能:
- 金額フォーマット
- パラメータバリデーション
- ログ設定
"""

from typing import Union
import logging


def setup_logger(name: str = 'simulator', level: int = logging.INFO) -> logging.Logger:
    """
    ロガーを設定
    
    Args:
        name: ロガー名
        level: ログレベル
        
    Returns:
        設定済みのロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def format_currency(
    value: Union[int, float],
    unit: str = '円',
    decimal_places: int = 0
) -> str:
    """
    金額を読みやすい形式にフォーマット
    
    Args:
        value: 金額
        unit: 単位（'円' or '万円'）
        decimal_places: 小数点以下桁数
        
    Returns:
        フォーマット済み文字列
    """
    if unit == '万円':
        value = value / 10000
    
    if decimal_places == 0:
        return f"{int(value):,}{unit}"
    else:
        return f"{value:,.{decimal_places}f}{unit}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    割合をパーセント表示にフォーマット
    
    Args:
        value: 割合（0.1 = 10%）
        decimal_places: 小数点以下桁数
        
    Returns:
        フォーマット済み文字列
    """
    return f"{value * 100:.{decimal_places}f}%"


def validate_positive(value: float, name: str) -> None:
    """
    正の値であることを検証
    
    Args:
        value: 検証する値
        name: パラメータ名（エラーメッセージ用）
        
    Raises:
        ValueError: 値が正でない場合
    """
    if value <= 0:
        raise ValueError(f"{name} は正の値である必要があります: {value}")


def validate_rate(value: float, name: str) -> None:
    """
    割合（0〜1）であることを検証
    
    Args:
        value: 検証する値
        name: パラメータ名（エラーメッセージ用）
        
    Raises:
        ValueError: 値が0〜1の範囲外の場合
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} は0〜1の範囲である必要があります: {value}")


def calculate_npv(
    cash_flows: list,
    discount_rate: float = 0.1
) -> float:
    """
    正味現在価値（NPV）を計算
    
    Args:
        cash_flows: キャッシュフローのリスト（初期投資は負の値）
        discount_rate: 割引率（年率）
        
    Returns:
        NPV
    """
    npv = 0
    monthly_rate = (1 + discount_rate) ** (1/12) - 1
    
    for t, cf in enumerate(cash_flows):
        npv += cf / ((1 + monthly_rate) ** t)
    
    return npv


def calculate_irr(
    cash_flows: list,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> float:
    """
    内部収益率（IRR）を計算
    
    Args:
        cash_flows: キャッシュフローのリスト
        max_iterations: 最大反復回数
        tolerance: 収束判定の許容誤差
        
    Returns:
        IRR（月次）
    """
    # 二分法による近似
    low, high = -0.99, 1.0
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        npv = sum(cf / ((1 + mid) ** t) for t, cf in enumerate(cash_flows))
        
        if abs(npv) < tolerance:
            return mid
        elif npv > 0:
            low = mid
        else:
            high = mid
    
    return mid


if __name__ == "__main__":
    # テスト
    print(format_currency(1234567, '円'))  # 1,234,567円
    print(format_currency(1234567, '万円', 1))  # 123.5万円
    print(format_percentage(0.087))  # 8.7%
    
    # NPV/IRRテスト
    cash_flows = [-1000000] + [50000] * 36
    print(f"NPV: {format_currency(calculate_npv(cash_flows), '円')}")
