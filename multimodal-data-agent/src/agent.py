"""
数据分析Agent核心模块：整合多模态解析、数据分析和可视化功能
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from .multimodal_parser import MultimodalParser
from .data_analyzer import DataAnalyzer
from .visualizer import Visualizer


class DataAnalysisAgent:
    """
    多模态数据分析Agent
    
    整合多模态内容理解、数据分析和可视化生成的全自动分析系统
    """
    
    def __init__(self, model_name: str = "default", verbose: bool = True):
        """
        初始化Agent
        
        Args:
            model_name: 使用的LLM模型名称
            verbose: 是否输出详细信息
        """
        self.model_name = model_name
        self.verbose = verbose
        
        # 初始化各组件
        self.parser = MultimodalParser()
        self.analyzer = DataAnalyzer()
        self.visualizer = Visualizer()
        
        # 状态管理
        self.current_data = None
        self.analysis_history = []
        self.context = {}
    
    def analyze(self, data_source: Union[str, Path, pd.DataFrame], 
                auto_visualize: bool = True) -> Dict[str, Any]:
        """
        分析数据源
        
        Args:
            data_source: 数据源（文件路径或DataFrame）
            auto_visualize: 是否自动生成可视化
            
        Returns:
            分析结果字典
        """
        if self.verbose:
            print(f"🔍 开始分析数据源: {data_source}")
        
        # 1. 加载和解析数据
        if isinstance(data_source, (str, Path)):
            file_path = Path(data_source)
            suffix = file_path.suffix.lower()
            
            if suffix in ['.csv', '.xlsx', '.xls', '.json', '.parquet']:
                self.current_data = self.parser.parse_tabular(file_path)
            elif suffix in ['.png', '.jpg', '.jpeg', '.bmp']:
                # 图像文件，尝试提取表格
                image_info = self.parser.parse_image(file_path)
                if image_info.get('structured_data') is not None:
                    self.current_data = image_info['structured_data']
                else:
                    return {
                        "error": "无法从图像中提取结构化数据",
                        "image_info": image_info
                    }
            else:
                return {"error": f"不支持的文件格式: {suffix}"}
        
        elif isinstance(data_source, pd.DataFrame):
            self.current_data = data_source.copy()
        
        else:
            return {"error": "无效的数据源类型"}
        
        if self.current_data is None or len(self.current_data) == 0:
            return {"error": "数据为空"}
        
        # 2. 加载到分析器
        self.analyzer.load_data(self.current_data)
        
        # 3. 获取数据摘要
        summary = self.analyzer.get_summary()
        
        # 4. 异常检测
        anomalies = self.analyzer.detect_anomalies()
        
        # 5. 自动生成可视化
        visualization_results = {}
        if auto_visualize:
            self.visualizer.set_data(self.current_data)
            visualization_results = self.visualizer.auto_visualize()
        
        # 6. 生成洞察报告
        insights = self._generate_insights(summary, anomalies)
        
        result = {
            "success": True,
            "data_shape": self.current_data.shape,
            "columns": list(self.current_data.columns),
            "summary": summary,
            "anomalies": anomalies,
            "visualizations": visualization_results,
            "insights": insights,
            "metadata": self.analyzer.metadata
        }
        
        # 记录历史
        self.analysis_history.append({
            "source": str(data_source),
            "timestamp": pd.Timestamp.now(),
            "result_summary": {
                "rows": self.current_data.shape[0],
                "cols": self.current_data.shape[1]
            }
        })
        
        if self.verbose:
            print(f"✅ 分析完成: {self.current_data.shape[0]}行 × {self.current_data.shape[1]}列")
            print(f"📊 发现 {len(insights)} 条关键洞察")
        
        return result
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        自然语言查询数据
        
        Args:
            question: 用户的问题
            
        Returns:
            查询结果
        """
        if self.current_data is None:
            return {"error": "请先加载数据"}
        
        if self.verbose:
            print(f"💬 处理问题: {question}")
        
        # 简单的规则匹配（实际应用中应使用LLM）
        question_lower = question.lower()
        
        result = {
            "question": question,
            "answer": "",
            "data": None,
            "chart_type": None
        }
        
        # 趋势分析
        if any(word in question_lower for word in ['趋势', 'trend', '变化', 'change']):
            numeric_cols = self.current_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                trend_result = self.analyzer.trend_analysis(
                    time_col=self.current_data.columns[0],
                    value_col=col
                )
                result["answer"] = f"{col}呈现{trend_result.get('trend_direction', '未知')}趋势"
                result["data"] = trend_result
                result["chart_type"] = "line"
        
        # 相关性分析
        elif any(word in question_lower for word in ['相关', 'correlation', '关系', 'relationship']):
            corr_matrix = self.analyzer.correlation_analysis()
            if len(corr_matrix) > 0:
                result["answer"] = "已生成变量相关性矩阵"
                result["data"] = corr_matrix.to_dict()
                result["chart_type"] = "heatmap"
        
        # 分布分析
        elif any(word in question_lower for word in ['分布', 'distribution', '统计', 'statistics']):
            summary = self.analyzer.get_summary()
            result["answer"] = "已生成数据统计摘要"
            result["data"] = summary
            result["chart_type"] = "histogram"
        
        # 异常值检测
        elif any(word in question_lower for word in ['异常', 'anomaly', '离群', 'outlier']):
            anomalies = self.analyzer.detect_anomalies()
            total = anomalies.get('total_anomalous_rows', 0)
            result["answer"] = f"检测到 {total} 行包含异常值"
            result["data"] = anomalies
            result["chart_type"] = "boxplot"
        
        # 分组分析
        elif any(word in question_lower for word in ['分组', 'group', '分类', 'category']):
            categorical_cols = self.current_data.select_dtypes(include=['object']).columns
            numeric_cols = self.current_data.select_dtypes(include=['number']).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                segment_result = self.analyzer.segment_analysis(
                    group_col=categorical_cols[0],
                    value_col=numeric_cols[0]
                )
                result["answer"] = f"按{categorical_cols[0]}分组的统计结果"
                result["data"] = segment_result
                result["chart_type"] = "bar"
        
        # 默认：返回数据摘要
        else:
            summary = self.analyzer.get_summary()
            result["answer"] = f"数据集包含{summary['shape'][0]}行{summary['shape'][1]}列数据"
            result["data"] = summary
            result["chart_type"] = "table"
        
        return result
    
    def multimodal_query(self, text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        多模态查询（文本+图像）
        
        Args:
            text: 文本问题
            image_path: 图像路径（可选）
            
        Returns:
            查询结果
        """
        result = {
            "text_query": text,
            "image_analyzed": False,
            "combined_result": {}
        }
        
        # 分析图像
        if image_path:
            try:
                image_info = self.parser.parse_image(image_path)
                result["image_analyzed"] = True
                result["image_content"] = image_info
                
                # 如果图像包含表格数据，合并到当前数据
                if image_info.get('structured_data') is not None:
                    img_df = image_info['structured_data']
                    if self.current_data is not None:
                        # 尝试合并数据
                        pass
                    else:
                        self.current_data = img_df
                        self.analyzer.load_data(self.current_data)
                
            except Exception as e:
                result["image_error"] = str(e)
        
        # 处理文本查询
        if self.current_data is not None:
            query_result = self.query(text)
            result["combined_result"] = query_result
        else:
            result["combined_result"] = {
                "answer": "没有可分析的数据",
                "suggestion": "请先上传数据文件或包含表格的图像"
            }
        
        return result
    
    def generate_report(self, output_path: Union[str, Path], 
                       format: str = "html") -> str:
        """
        生成完整分析报告
        
        Args:
            output_path: 输出路径
            format: 输出格式 ("html" 或 "json")
            
        Returns:
            输出文件路径
        """
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        output_path = Path(output_path)
        
        # 收集所有分析结果
        report_data = {
            "title": "数据分析报告",
            "generated_at": pd.Timestamp.now().isoformat(),
            "data_overview": {
                "shape": self.current_data.shape,
                "columns": list(self.current_data.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.current_data.dtypes.items()}
            },
            "summary": self.analyzer.get_summary(),
            "anomalies": self.analyzer.detect_anomalies(),
            "insights": self._generate_insights(
                self.analyzer.get_summary(),
                self.analyzer.detect_anomalies()
            )
        }
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        elif format == "html":
            # 生成可视化仪表板
            dashboard_path = output_path.with_name(output_path.stem + "_dashboard.html")
            self.visualizer.set_data(self.current_data)
            self.visualizer.auto_visualize()
            self.visualizer.create_dashboard(dashboard_path)
            
            # 生成HTML报告
            html_content = self._generate_html_report(report_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        if self.verbose:
            print(f"📄 报告已保存至: {output_path}")
        
        return str(output_path)
    
    def _generate_insights(self, summary: Dict, anomalies: Dict) -> List[str]:
        """生成数据洞察"""
        insights = []
        
        # 数据质量洞察
        missing_total = sum(summary.get('missing_values', {}).values())
        if missing_total > 0:
            missing_ratio = missing_total / (summary['shape'][0] * summary['shape'][1]) * 100
            insights.append(f"数据缺失率为{missing_ratio:.2f}%，建议检查数据采集流程")
        
        # 数值特征洞察
        for col, stats in summary.get('numeric_stats', {}).items():
            if stats and stats.get('std'):
                cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
                if cv > 1:
                    insights.append(f"{col}的变异系数较高({cv:.2f})，数据波动较大")
        
        # 异常值洞察
        total_anomalies = anomalies.get('total_anomalous_rows', 0)
        if total_anomalies > 0:
            anomaly_ratio = total_anomalies / summary['shape'][0] * 100
            insights.append(f"检测到{total_anomalies}行异常数据({anomaly_ratio:.2f}%)，建议进一步审查")
        
        # 通用建议
        if summary['shape'][0] < 100:
            insights.append("数据量较小，统计结果可能不够稳定")
        
        if len(summary.get('categorical_stats', {})) > 5:
            insights.append("分类变量较多，可考虑进行特征编码后建模")
        
        return insights
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """生成HTML格式报告"""
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>{report_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
                .insight {{ background: #e8f6f3; padding: 15px; margin: 10px 0; border-left: 4px solid #1abc9c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 {report_data['title']}</h1>
            <p>生成时间: {report_data['generated_at']}</p>
            
            <h2>数据概览</h2>
            <div class="metric">行数: <strong>{report_data['data_overview']['shape'][0]}</strong></div>
            <div class="metric">列数: <strong>{report_data['data_overview']['shape'][1]}</strong></div>
            
            <h2>关键洞察</h2>
            {''.join([f'<div class="insight">✓ {insight}</div>' for insight in report_data['insights']])}
            
            <h2>数据摘要</h2>
            <p>详见附带的交互式仪表板</p>
            
            <h2>异常检测</h2>
            <p>检测到异常行数：<strong>{report_data['anomalies'].get('total_anomalous_rows', 0)}</strong></p>
        </body>
        </html>
        """
        return html
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前分析上下文"""
        return self.context.copy()
    
    def clear(self):
        """清除当前状态"""
        self.current_data = None
        self.context = {}
        self.visualizer.clear_figures()
        if self.verbose:
            print("🔄 已清除当前状态")
