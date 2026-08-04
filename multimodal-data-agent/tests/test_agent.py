"""
测试模块：验证数据分析Agent的各项功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DataAnalysisAgent
from src.multimodal_parser import MultimodalParser
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer


class TestMultimodalParser:
    """测试多模态解析器"""
    
    def setup_method(self):
        """测试前准备"""
        self.parser = MultimodalParser()
        self.test_dir = Path(__file__).parent.parent / "data"
        self.test_dir.mkdir(exist_ok=True)
        
        # 创建测试CSV文件
        self.test_csv = self.test_dir / "test_data.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'salary': [50000, 60000, 70000, 80000],
            'department': ['HR', 'Engineering', 'Sales', 'Marketing']
        })
        df.to_csv(self.test_csv, index=False)
    
    def test_parse_csv(self):
        """测试CSV文件解析"""
        df = self.parser.parse_tabular(self.test_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.columns) == ['name', 'age', 'salary', 'department']
    
    def test_parse_excel(self):
        """测试Excel文件解析"""
        test_excel = self.test_dir / "test_data.xlsx"
        df_original = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_original.to_excel(test_excel, index=False)
        
        df = self.parser.parse_tabular(test_excel)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_get_image_info(self):
        """测试图像信息获取"""
        from PIL import Image
        
        # 创建测试图像
        test_img = self.test_dir / "test_image.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_img)
        
        info = self.parser.get_image_info(test_img)
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'PNG'


class TestDataAnalyzer:
    """测试数据分析器"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = DataAnalyzer()
        
        # 创建测试数据
        np.random.seed(42)
        self.df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'sales': np.random.randint(1000, 5000, 100),
            'profit': np.random.randint(100, 1000, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
    
    def test_load_data(self):
        """测试数据加载"""
        result = self.analyzer.load_data(self.df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
    
    def test_get_summary(self):
        """测试数据摘要"""
        self.analyzer.load_data(self.df)
        summary = self.analyzer.get_summary()
        
        assert 'shape' in summary
        assert summary['shape'][0] == 100
        assert 'numeric_stats' in summary
        assert 'sales' in summary['numeric_stats']
    
    def test_detect_anomalies(self):
        """测试异常检测"""
        # 添加一些异常值
        df_with_outliers = self.df.copy()
        df_with_outliers.loc[0, 'sales'] = 100000  # 异常值
        
        self.analyzer.load_data(df_with_outliers)
        anomalies = self.analyzer.detect_anomalies()
        
        assert 'anomalies' in anomalies
        assert 'sales' in anomalies['anomalies']
    
    def test_correlation_analysis(self):
        """测试相关性分析"""
        self.analyzer.load_data(self.df)
        corr_matrix = self.analyzer.correlation_analysis()
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
    
    def test_trend_analysis(self):
        """测试趋势分析"""
        self.analyzer.load_data(self.df)
        trend = self.analyzer.trend_analysis('date', 'sales')
        
        assert 'trend_direction' in trend
        assert trend['trend_direction'] in ['increasing', 'decreasing']


class TestVisualizer:
    """测试可视化生成器"""
    
    def setup_method(self):
        """测试前准备"""
        self.visualizer = Visualizer(style="matplotlib")  # 使用matplotlib避免plotly依赖问题
        
        # 创建测试数据
        np.random.seed(42)
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'] * 25,
            'value': np.random.randn(100),
            'value2': np.random.randn(100)
        })
    
    def test_create_histogram(self):
        """测试直方图创建"""
        self.visualizer.set_data(self.df)
        fig = self.visualizer.create_histogram('value')
        assert fig is not None
    
    def test_create_bar_chart(self):
        """测试条形图创建"""
        self.visualizer.set_data(self.df)
        fig = self.visualizer.create_bar_chart('category')
        assert fig is not None
    
    def test_auto_visualize(self):
        """测试自动可视化"""
        results = self.visualizer.auto_visualize(self.df)
        
        assert 'charts' in results
        assert 'recommendations' in results
        assert len(results['charts']) > 0


class TestDataAnalysisAgent:
    """测试数据分析Agent"""
    
    def setup_method(self):
        """测试前准备"""
        self.agent = DataAnalysisAgent(verbose=False)
        
        # 创建测试数据
        self.test_dir = Path(__file__).parent.parent / "data"
        self.test_dir.mkdir(exist_ok=True)
        
        np.random.seed(42)
        self.df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'revenue': np.random.randint(10000, 50000, 50),
            'costs': np.random.randint(5000, 30000, 50),
            'product': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        self.test_csv = self.test_dir / "agent_test.csv"
        self.df.to_csv(self.test_csv, index=False)
    
    def teardown_method(self):
        """测试后清理"""
        if self.test_csv.exists():
            self.test_csv.unlink()
    
    def test_analyze_dataframe(self):
        """测试DataFrame分析"""
        result = self.agent.analyze(self.df)
        
        assert result['success'] == True
        assert result['data_shape'][0] == 50
        assert result['data_shape'][1] == 4
        assert 'summary' in result
        assert 'insights' in result
    
    def test_analyze_file(self):
        """测试文件分析"""
        result = self.agent.analyze(self.test_csv)
        
        assert result['success'] == True
        assert result['data_shape'][0] == 50
    
    def test_query(self):
        """测试自然语言查询"""
        self.agent.analyze(self.df)
        
        # 测试趋势查询
        result = self.agent.query("销售额的趋势如何？")
        assert 'answer' in result
        
        # 测试统计查询
        result = self.agent.query("数据的统计分布")
        assert 'answer' in result
    
    def test_generate_report(self):
        """测试报告生成"""
        self.agent.analyze(self.df)
        
        output_path = self.test_dir / "test_report.html"
        report_path = self.agent.generate_report(output_path, format="html")
        
        assert Path(report_path).exists()
        assert Path(report_path).stat().st_size > 0
    
    def test_multimodal_query(self):
        """测试多模态查询"""
        self.agent.analyze(self.df)
        
        result = self.agent.multimodal_query("分析销售数据")
        assert 'text_query' in result
        assert 'combined_result' in result
    
    def test_clear_state(self):
        """测试状态清除"""
        self.agent.analyze(self.df)
        assert self.agent.current_data is not None
        
        self.agent.clear()
        assert self.agent.current_data is None


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.test_dir = Path(__file__).parent.parent / "data"
        self.test_dir.mkdir(exist_ok=True)
        
        # 创建综合测试数据
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            'order_id': range(1, n+1),
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
            'quantity': np.random.randint(1, 10, n),
            'unit_price': np.random.uniform(10, 500, n),
            'customer_age': np.random.randint(18, 70, n),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n),
            'rating': np.random.uniform(1, 5, n)
        })
        
        # 计算总金额
        self.df['total_amount'] = self.df['quantity'] * self.df['unit_price']
        
        self.test_csv = self.test_dir / "sales_data.csv"
        self.df.to_csv(self.test_csv, index=False)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        agent = DataAnalysisAgent(verbose=False)
        
        # 1. 分析数据
        result = agent.analyze(self.test_csv)
        assert result['success'] == True
        
        # 2. 查询洞察
        query_result = agent.query("各产品类别的销售情况")
        assert 'answer' in query_result
        
        # 3. 生成报告
        report_path = self.test_dir / "full_analysis_report.html"
        agent.generate_report(report_path)
        assert Path(report_path).exists()
        
        # 4. 验证输出
        assert result['data_shape'][0] == 200
        assert 'total_amount' in result['columns']
        assert len(result['insights']) > 0
    
    def test_error_handling(self):
        """测试错误处理"""
        agent = DataAnalysisAgent(verbose=False)
        
        # 测试空查询
        result = agent.query("test")
        assert 'error' in result
        
        # 测试无效文件
        result = agent.analyze("/nonexistent/file.csv")
        assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
