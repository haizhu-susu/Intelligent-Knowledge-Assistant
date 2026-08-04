"""
演示脚本：展示数据分析Agent的完整功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import DataAnalysisAgent


def create_sample_data():
    """创建示例销售数据"""
    np.random.seed(42)
    n = 365
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n),
        'product_category': np.random.choice(
            ['Electronics', 'Clothing', 'Food', 'Books', 'Home'], 
            n,
            p=[0.25, 0.20, 0.30, 0.15, 0.10]
        ),
        'region': np.random.choice(
            ['North', 'South', 'East', 'West', 'Central'], 
            n
        ),
        'quantity': np.random.randint(1, 20, n),
        'unit_price': np.round(np.random.uniform(10, 500, n), 2),
        'customer_age': np.random.randint(18, 70, n),
        'customer_gender': np.random.choice(['Male', 'Female'], n),
        'rating': np.round(np.random.uniform(1, 5, n), 1),
        'is_member': np.random.choice([True, False], n, p=[0.4, 0.6])
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    df['discount'] = np.where(df['is_member'], 
                              np.round(df['total_amount'] * 0.1, 2), 
                              0)
    df['final_amount'] = df['total_amount'] - df['discount']
    
    return df


def demo_basic_analysis():
    """基础分析演示"""
    print("=" * 60)
    print("🎯 演示 1: 基础数据分析")
    print("=" * 60)
    
    # 创建Agent
    agent = DataAnalysisAgent(verbose=True)
    
    # 生成示例数据
    df = create_sample_data()
    
    # 分析数据
    result = agent.analyze(df)
    
    # 显示结果
    print(f"\n📊 数据形状：{result['data_shape']}")
    print(f"📋 列名：{', '.join(result['columns'])}")
    
    print("\n💡 关键洞察:")
    for i, insight in enumerate(result['insights'], 1):
        print(f"  {i}. {insight}")
    
    return agent


def demo_natural_language_query(agent):
    """自然语言查询演示"""
    print("\n" + "=" * 60)
    print("💬 演示 2: 自然语言查询")
    print("=" * 60)
    
    queries = [
        "数据的统计分布如何？",
        "销售额的趋势是什么？",
        "有哪些异常值？",
        "各产品类别的销售情况"
    ]
    
    for query in queries:
        print(f"\n❓ 问题：{query}")
        result = agent.query(query)
        print(f"💡 回答：{result['answer']}")


def demo_visualization(agent):
    """可视化演示"""
    print("\n" + "=" * 60)
    print("📈 演示 3: 自动可视化")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 生成可视化
    viz_result = agent.visualizer.auto_visualize(agent.current_data)
    
    print(f"\n✅ 生成了 {len(viz_result['charts'])} 个图表:")
    for chart in viz_result['charts'][:5]:
        print(f"  - {chart['type']}: {chart.get('description', '')}")
    
    # 创建仪表板
    dashboard_path = output_dir / "dashboard.html"
    agent.visualizer.create_dashboard(dashboard_path)
    print(f"\n📊 交互式仪表板已保存至：{dashboard_path}")
    
    # 建议
    print("\n💡 可视化建议:")
    for rec in viz_result['recommendations']:
        print(f"  - {rec}")


def demo_report_generation(agent):
    """报告生成演示"""
    print("\n" + "=" * 60)
    print("📄 演示 4: 自动生成报告")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "analysis_report.html"
    agent.generate_report(report_path, format="html")
    
    print(f"\n✅ 分析报告已生成:")
    print(f"  📄 主报告：{report_path}")
    print(f"  📊 仪表板：{output_dir / 'analysis_report_dashboard.html'}")


def demo_multimodal():
    """多模态功能演示"""
    print("\n" + "=" * 60)
    print("🖼️ 演示 5: 多模态分析能力")
    print("=" * 60)
    
    agent = DataAnalysisAgent(verbose=False)
    
    # 创建示例图像（实际应用中可以是截图）
    from PIL import Image
    import io
    
    # 创建一个简单的表格图像示意
    img = Image.new('RGB', (400, 200), color='white')
    img_path = Path(__file__).parent / "data" / "sample_table.png"
    img_path.parent.mkdir(exist_ok=True)
    img.save(img_path)
    
    # 多模态查询
    result = agent.multimodal_query(
        text="分析这个数据表格",
        image_path=str(img_path)
    )
    
    print(f"\n🖼️ 图像分析：{'成功' if result['image_analyzed'] else '失败'}")
    print(f"💬 文本查询：{result['text_query']}")
    print(f"📊 结果：{result['combined_result'].get('answer', '无数据')}")


def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("多模态数据分析可视化 Agent 演示")
    print("🚀" * 30 + "\n")
    
    try:
        # 1. 基础分析
        agent = demo_basic_analysis()
        
        # 2. 自然语言查询
        demo_natural_language_query(agent)
        
        # 3. 可视化
        demo_visualization(agent)
        
        # 4. 报告生成
        demo_report_generation(agent)
        
        # 5. 多模态功能
        demo_multimodal()
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)
        print("\n📂 输出文件位置:")
        print(f"   {Path(__file__).parent / 'output'}")
        print("\n💡 提示:")
        print("   - 打开 dashboard.html 查看交互式图表")
        print("   - 打开 analysis_report.html 查看完整报告")
        print("   - 修改 demo.py 中的数据源进行自定义分析")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
