"""
Streamlit 前端应用：交互式数据分析界面
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import DataAnalysisAgent


def main():
    """主函数"""
    st.set_page_config(
        page_title="多模态数据分析 Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 多模态数据分析可视化 Agent")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        upload_option = st.radio(
            "数据输入方式",
            ["上传文件", "使用示例数据"]
        )
        
        verbose = st.checkbox("显示详细信息", value=True)
        
        st.markdown("---")
        st.info("""
        **支持的文件格式:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        - 图像中的表格 (.png, .jpg)
        """)
    
    # 初始化Agent
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalysisAgent(verbose=verbose)
    
    agent = st.session_state.agent
    
    # 数据加载
    df = None
    
    if upload_option == "上传文件":
        uploaded_file = st.file_uploader(
            "选择数据文件",
            type=['csv', 'xlsx', 'xls', 'json']
        )
        
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                if file_type == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_type in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_type == 'json':
                    df = pd.read_json(uploaded_file)
                
                st.success(f"✅ 成功加载 {len(df)} 行数据")
                
            except Exception as e:
                st.error(f"❌ 加载失败：{e}")
    
    else:
        # 使用示例数据
        if st.button("生成示例销售数据"):
            np.random.seed(42)
            n = 365
            
            df = pd.DataFrame({
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
                'rating': np.round(np.random.uniform(1, 5, n), 1)
            })
            
            df['total_amount'] = df['quantity'] * df['unit_price']
            st.success(f"✅ 生成了 {len(df)} 行示例数据")
    
    # 如果有数据，进行分析
    if df is not None:
        # 显示原始数据
        with st.expander("📊 查看原始数据", expanded=False):
            st.dataframe(df, use_container_width=True)
        
        # 分析按钮
        if st.button("🔍 开始智能分析", type="primary"):
            with st.spinner("正在分析数据..."):
                # 执行分析
                result = agent.analyze(df, auto_visualize=True)
                
                # 保存结果到session_state
                st.session_state.analysis_result = result
        
        # 显示分析结果
        if hasattr(st.session_state, 'analysis_result'):
            result = st.session_state.analysis_result
            
            # 关键指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("行数", result['data_shape'][0])
            with col2:
                st.metric("列数", result['data_shape'][1])
            with col3:
                st.metric("数值列", len(result['summary'].get('numeric_stats', {})))
            with col4:
                st.metric("分类列", len(result['summary'].get('categorical_stats', {})))
            
            st.markdown("---")
            
            # 洞察和建议
            st.subheader("💡 关键洞察")
            for insight in result['insights']:
                st.info(insight)
            
            # 自然语言查询
            st.markdown("---")
            st.subheader("💬 自然语言查询")
            
            query = st.text_input(
                "输入您的问题",
                placeholder="例如：数据的统计分布如何？销售额的趋势是什么？"
            )
            
            if query:
                with st.spinner("正在处理问题..."):
                    query_result = agent.query(query)
                    st.write(f"**回答:** {query_result['answer']}")
            
            # 可视化
            st.markdown("---")
            st.subheader("📈 可视化图表")
            
            viz_cols = st.columns(2)
            
            if result['visualizations'].get('charts'):
                for i, chart in enumerate(result['visualizations']['charts'][:6]):
                    col_idx = i % 2
                    
                    with viz_cols[col_idx]:
                        st.markdown(f"**{chart['type']}**: {chart.get('description', '')}")
                        
                        # 根据图表类型创建对应的plotly图
                        chart_type = chart['type']
                        
                        if chart_type == 'histogram':
                            col = chart.get('column')
                            if col:
                                fig = agent.visualizer.create_histogram(col)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == 'bar_chart':
                            col = chart.get('column')
                            if col:
                                fig = agent.visualizer.create_bar_chart(col)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == 'correlation_heatmap':
                            fig = agent.visualizer.create_correlation_heatmap()
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            
            # 导出选项
            st.markdown("---")
            st.subheader("📥 导出报告")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("下载 HTML 报告"):
                    output_path = Path(__file__).parent / "output" / "streamlit_report.html"
                    output_path.parent.mkdir(exist_ok=True)
                    agent.generate_report(output_path, format="html")
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="点击下载",
                            data=f.read(),
                            file_name="analysis_report.html",
                            mime="text/html"
                        )
            
            with export_col2:
                if st.button("下载 JSON 数据"):
                    output_path = Path(__file__).parent / "output" / "analysis_data.json"
                    output_path.parent.mkdir(exist_ok=True)
                    agent.generate_report(output_path, format="json")
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="点击下载",
                            data=f.read(),
                            file_name="analysis_data.json",
                            mime="application/json"
                        )
    
    else:
        # 没有数据时的提示
        st.info("👆 请在左侧上传数据文件或生成示例数据开始分析")
        
        # 展示功能介绍
        st.markdown("""
        ### 🎯 功能特性
        
        - **智能数据理解**: 自动识别数据类型和特征
        - **统计分析**: 描述性统计、相关性分析、异常检测
        - **趋势分析**: 时间序列趋势识别和预测
        - **自动可视化**: 根据数据特征生成最优图表
        - **自然语言交互**: 通过对话方式探索数据
        - **报告生成**: 一键生成完整分析报告
        
        ### 📊 支持的图表类型
        
        - 直方图 (Histogram)
        - 条形图 (Bar Chart)
        - 散点图 (Scatter Plot)
        - 箱线图 (Box Plot)
        - 热力图 (Heatmap)
        - 折线图 (Line Chart)
        """)


if __name__ == "__main__":
    main()
