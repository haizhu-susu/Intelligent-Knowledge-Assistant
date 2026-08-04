"""
可视化生成器：自动创建各种类型的图表和仪表板
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """可视化生成器"""
    
    def __init__(self, style: str = "plotly"):
        """
        初始化可视化生成器
        
        Args:
            style: 可视化风格 ("plotly" 或 "matplotlib")
        """
        self.style = style
        self.figures = []
        self.current_df = None
        
        # 设置默认样式
        if style == "matplotlib":
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def set_data(self, df: pd.DataFrame):
        """设置要可视化的数据"""
        self.current_df = df.copy()
    
    def auto_visualize(self, df: Optional[pd.DataFrame] = None, 
                      output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        自动根据数据特征生成合适的可视化
        
        Args:
            df: 数据DataFrame
            output_path: 输出路径（可选）
            
        Returns:
            生成的图表信息
        """
        if df is not None:
            self.current_df = df.copy()
        
        if self.current_df is None:
            return {"error": "No data provided"}
        
        results = {
            "charts": [],
            "recommendations": []
        }
        
        # 分析数据类型
        numeric_cols = self.current_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.current_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 尝试检测日期列
        for col in self.current_df.columns:
            if col not in datetime_cols:
                try:
                    pd.to_datetime(self.current_df[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        # 1. 数值列分布直方图
        for col in numeric_cols[:5]:  # 限制数量
            fig = self.create_histogram(col)
            if fig:
                results["charts"].append({
                    "type": "histogram",
                    "column": col,
                    "description": f"{col}的分布情况"
                })
        
        # 2. 分类变量条形图
        for col in categorical_cols[:3]:
            if self.current_df[col].nunique() <= 20:  # 类别不宜过多
                fig = self.create_bar_chart(col)
                if fig:
                    results["charts"].append({
                        "type": "bar_chart",
                        "column": col,
                        "description": f"{col}的频数分布"
                    })
        
        # 3. 数值变量相关性热力图
        if len(numeric_cols) >= 2:
            fig = self.create_correlation_heatmap()
            if fig:
                results["charts"].append({
                    "type": "correlation_heatmap",
                    "description": "数值变量相关性矩阵"
                })
        
        # 4. 时间序列趋势图
        if datetime_cols and numeric_cols:
            time_col = datetime_cols[0]
            for value_col in numeric_cols[:2]:
                fig = self.create_time_series(time_col, value_col)
                if fig:
                    results["charts"].append({
                        "type": "time_series",
                        "time_column": time_col,
                        "value_column": value_col,
                        "description": f"{value_col}随时间的变化趋势"
                    })
        
        # 5. 散点图（关系分析）
        if len(numeric_cols) >= 2:
            fig = self.create_scatter_plot(numeric_cols[0], numeric_cols[1])
            if fig:
                results["charts"].append({
                    "type": "scatter_plot",
                    "x_column": numeric_cols[0],
                    "y_column": numeric_cols[1],
                    "description": f"{numeric_cols[0]}与{numeric_cols[1]}的关系"
                })
        
        # 6. 箱线图（异常值检测）
        for col in numeric_cols[:3]:
            fig = self.create_box_plot(col)
            if fig:
                results["charts"].append({
                    "type": "box_plot",
                    "column": col,
                    "description": f"{col}的分布及异常值"
                })
        
        # 生成建议
        results["recommendations"] = self._generate_recommendations(
            numeric_cols, categorical_cols, datetime_cols
        )
        
        # 保存输出
        if output_path:
            self.save_all_charts(output_path)
        
        return results
    
    def create_histogram(self, column: str, bins: int = 30) -> Optional[go.Figure]:
        """创建直方图"""
        if self.current_df is None or column not in self.current_df.columns:
            return None
        
        if self.style == "plotly":
            fig = px.histogram(
                self.current_df, 
                x=column, 
                nbins=bins,
                title=f"{column} Distribution",
                template="plotly_white"
            )
            fig.update_layout(showlegend=False)
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.current_df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{column} Distribution')
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_bar_chart(self, column: str, top_n: int = 10) -> Optional[go.Figure]:
        """创建条形图"""
        if self.current_df is None or column not in self.current_df.columns:
            return None
        
        value_counts = self.current_df[column].value_counts().head(top_n)
        
        if self.style == "plotly":
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top {top_n} {column}",
                labels={'x': 'Count', 'y': column},
                template="plotly_white"
            )
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(value_counts)), value_counts.values)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels(value_counts.index)
            ax.set_xlabel('Count')
            ax.set_title(f'Top {top_n} {column}')
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_correlation_heatmap(self) -> Optional[go.Figure]:
        """创建相关性热力图"""
        if self.current_df is None:
            return None
        
        numeric_df = self.current_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        
        corr_matrix = numeric_df.corr()
        
        if self.style == "plotly":
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r'
            )
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_time_series(self, time_col: str, value_col: str) -> Optional[go.Figure]:
        """创建时间序列图"""
        if self.current_df is None:
            return None
        
        if time_col not in self.current_df.columns or value_col not in self.current_df.columns:
            return None
        
        df_copy = self.current_df.copy()
        try:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            df_sorted = df_copy.sort_values(time_col)
        except:
            return None
        
        if self.style == "plotly":
            fig = px.line(
                df_sorted,
                x=time_col,
                y=value_col,
                title=f"{value_col} Over Time",
                template="plotly_white"
            )
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_sorted[time_col], df_sorted[value_col], marker='o', markersize=3)
            ax.set_xlabel(time_col)
            ax.set_ylabel(value_col)
            ax.set_title(f'{value_col} Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_scatter_plot(self, x_col: str, y_col: str, 
                           color_col: Optional[str] = None) -> Optional[go.Figure]:
        """创建散点图"""
        if self.current_df is None:
            return None
        
        if x_col not in self.current_df.columns or y_col not in self.current_df.columns:
            return None
        
        if self.style == "plotly":
            fig = px.scatter(
                self.current_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col else None,
                title=f"{x_col} vs {y_col}",
                template="plotly_white",
                hover_data=self.current_df.columns.tolist()
            )
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                self.current_df[x_col], 
                self.current_df[y_col],
                c=self.current_df[color_col] if color_col else 'blue',
                alpha=0.6
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{x_col} vs {y_col}')
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_box_plot(self, column: str, group_col: Optional[str] = None) -> Optional[go.Figure]:
        """创建箱线图"""
        if self.current_df is None or column not in self.current_df.columns:
            return None
        
        if self.style == "plotly":
            if group_col and group_col in self.current_df.columns:
                fig = px.box(
                    self.current_df,
                    x=group_col,
                    y=column,
                    title=f"{column} by {group_col}",
                    template="plotly_white"
                )
            else:
                fig = px.box(
                    self.current_df,
                    y=column,
                    title=f"{column} Distribution",
                    template="plotly_white"
                )
            self.figures.append(fig)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            if group_col and group_col in self.current_df.columns:
                self.current_df.boxplot(column=column, by=group_col, ax=ax)
            else:
                self.current_df.boxplot(column=column, ax=ax)
            plt.title(f'{column} Box Plot')
            plt.suptitle('')
            plt.tight_layout()
            self.figures.append(fig)
            return fig
    
    def create_dashboard(self, output_path: Union[str, Path]) -> str:
        """
        创建交互式仪表板
        
        Args:
            output_path: HTML输出路径
            
        Returns:
            输出文件路径
        """
        if not self.figures:
            raise ValueError("No figures to display. Generate charts first.")
        
        output_path = Path(output_path)
        
        if self.style == "plotly":
            from plotly.io import to_html
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Analysis Dashboard</title>
                <meta charset="utf-8" />
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .chart-container {{ margin: 20px 0; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>📊 Data Analysis Dashboard</h1>
            """
            
            for i, fig in enumerate(self.figures):
                chart_html = to_html(fig, full_html=False, include_plotlyjs='cdn')
                html_content += f'<div class="chart-container">{chart_html}</div>'
            
            html_content += """
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        else:
            # Matplotlib风格
            fig_count = len(self.figures)
            cols = 2
            rows = (fig_count + 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            for i, chart_fig in enumerate(self.figures):
                if i < len(axes):
                    # 复制图表到子图
                    pass
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return str(output_path)
    
    def save_all_charts(self, base_path: Union[str, Path], format: str = "png"):
        """保存所有图表"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            if self.style == "plotly":
                file_path = base_path / f"chart_{i+1}.{format}"
                if format == "html":
                    fig.write_html(str(file_path))
                else:
                    fig.write_image(str(file_path))
            else:
                file_path = base_path / f"chart_{i+1}.png"
                fig.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    def _generate_recommendations(self, numeric_cols: List[str], 
                                 categorical_cols: List[str],
                                 datetime_cols: List[str]) -> List[str]:
        """生成可视化建议"""
        recommendations = []
        
        if len(numeric_cols) >= 2:
            recommendations.append("考虑使用散点图矩阵探索数值变量间的关系")
        
        if datetime_cols and numeric_cols:
            recommendations.append("使用时间序列图展示指标随时间的变化趋势")
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append("使用分组箱线图比较不同类别的数值分布")
        
        if len(numeric_cols) >= 3:
            recommendations.append("考虑使用PCA降维后进行可视化")
        
        return recommendations
    
    def clear_figures(self):
        """清除所有图表"""
        self.figures = []
        if self.style == "matplotlib":
            plt.close('all')
