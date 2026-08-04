"""
数据分析模块：提供统计分析、数据清洗、特征工程等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json


class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self):
        """初始化数据分析器"""
        self.df = None
        self.metadata = {}
        self.analysis_cache = {}
    
    def load_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data: DataFrame或文件路径
            
        Returns:
            加载后的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, (str, Path)):
            parser = MultimodalParser()
            self.df = parser.parse_tabular(data)
        else:
            raise ValueError("Data must be DataFrame or file path")
        
        self._compute_metadata()
        return self.df
    
    def _compute_metadata(self):
        """计算数据元信息"""
        if self.df is None:
            return
        
        self.metadata = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取数据摘要统计
        
        Returns:
            包含统计信息的字典
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "shape": self.df.shape,
            "columns": self.metadata["column_names"],
            "dtypes": self.metadata["dtypes"],
            "numeric_stats": {},
            "categorical_stats": {},
            "missing_values": self.metadata["null_counts"]
        }
        
        # 数值型列统计
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary["numeric_stats"][col] = {
                "count": int(self.df[col].count()),
                "mean": float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                "std": float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None,
                "min": float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                "max": float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                "median": float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None,
                "quantiles": {
                    "25%": float(self.df[col].quantile(0.25)),
                    "50%": float(self.df[col].quantile(0.50)),
                    "75%": float(self.df[col].quantile(0.75))
                }
            }
        
        # 分类型列统计
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = self.df[col].value_counts().head(10)
            summary["categorical_stats"][col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_values": value_counts.to_dict(),
                "missing_ratio": float(self.df[col].isnull().mean())
            }
        
        return summary
    
    def detect_anomalies(self, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
        """
        检测异常值
        
        Args:
            method: 检测方法 ("iqr" 或 "zscore")
            threshold: 阈值
            
        Returns:
            异常值检测结果
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        anomalies = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                anomaly_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == "zscore":
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = np.abs((self.df[col] - mean) / std)
                anomaly_mask = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            anomaly_indices = self.df[anomaly_mask].index.tolist()
            anomaly_values = self.df.loc[anomaly_mask, col].tolist()
            
            if anomaly_indices:
                anomalies[col] = {
                    "count": len(anomaly_indices),
                    "indices": anomaly_indices[:20],  # 限制返回数量
                    "values": anomaly_values[:20],
                    "percentage": len(anomaly_indices) / len(self.df) * 100
                }
        
        return {
            "method": method,
            "threshold": threshold,
            "anomalies": anomalies,
            "total_anomalous_rows": len(set([idx for col_data in anomalies.values() 
                                             for idx in col_data["indices"]]))
        }
    
    def correlation_analysis(self, method: str = "pearson") -> pd.DataFrame:
        """
        相关性分析
        
        Args:
            method: 相关系数计算方法
            
        Returns:
            相关系数矩阵
        """
        if self.df is None:
            return pd.DataFrame()
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    
    def trend_analysis(self, time_col: str, value_col: str) -> Dict[str, Any]:
        """
        趋势分析
        
        Args:
            time_col: 时间列名
            value_col: 数值列名
            
        Returns:
            趋势分析结果
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        if time_col not in self.df.columns or value_col not in self.df.columns:
            return {"error": "Specified columns not found"}
        
        # 尝试转换时间列
        df_copy = self.df.copy()
        try:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        except:
            return {"error": "Cannot convert time column to datetime"}
        
        # 按时间排序
        df_sorted = df_copy.sort_values(time_col)
        
        # 计算移动平均
        window = min(7, len(df_sorted) // 4)
        if window > 0:
            df_sorted['moving_avg'] = df_sorted[value_col].rolling(window=window).mean()
        
        # 计算增长率
        df_sorted['growth_rate'] = df_sorted[value_col].pct_change()
        
        # 趋势方向
        recent_trend = df_sorted[value_col].tail(len(df_sorted)//4).mean()
        early_trend = df_sorted[value_col].head(len(df_sorted)//4).mean()
        
        trend_direction = "increasing" if recent_trend > early_trend else "decreasing"
        
        return {
            "trend_direction": trend_direction,
            "average_growth_rate": float(df_sorted['growth_rate'].mean()) if not pd.isna(df_sorted['growth_rate'].mean()) else 0,
            "volatility": float(df_sorted[value_col].std()),
            "data_points": len(df_sorted),
            "time_range": {
                "start": str(df_sorted[time_col].min()),
                "end": str(df_sorted[time_col].max())
            }
        }
    
    def segment_analysis(self, group_col: str, value_col: str) -> Dict[str, Any]:
        """
        分组/细分分析
        
        Args:
            group_col: 分组列名
            value_col: 数值列名
            
        Returns:
            分组统计结果
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        grouped = self.df.groupby(group_col)[value_col]
        
        stats = grouped.agg(['count', 'mean', 'std', 'min', 'max', 'median'])
        stats = stats.reset_index()
        
        # 转换为字典格式
        result = {
            "groups": stats.to_dict('records'),
            "total_groups": self.df[group_col].nunique(),
            "largest_group": self.df[group_col].value_counts().idxmax(),
            "smallest_group": self.df[group_col].value_counts().idxmin()
        }
        
        return result
    
    def feature_importance(self, target_col: str, method: str = "correlation") -> List[Dict]:
        """
        特征重要性分析（简化版）
        
        Args:
            target_col: 目标变量列名
            method: 分析方法
            
        Returns:
            特征重要性列表
        """
        if self.df is None:
            return []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in numeric_cols:
            return []
        
        correlations = []
        for col in numeric_cols:
            if col != target_col:
                corr = self.df[[col, target_col]].corr().iloc[0, 1]
                correlations.append({
                    "feature": col,
                    "correlation": float(corr) if not pd.isna(corr) else 0,
                    "abs_correlation": abs(float(corr)) if not pd.isna(corr) else 0
                })
        
        # 按绝对值排序
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return correlations
    
    def export_insights(self, output_path: Union[str, Path]) -> str:
        """
        导出分析洞察
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        insights = {
            "summary": self.get_summary(),
            "anomalies": self.detect_anomalies(),
            "metadata": self.metadata
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        
        return str(output_path)


# 延迟导入避免循环依赖
from .multimodal_parser import MultimodalParser
