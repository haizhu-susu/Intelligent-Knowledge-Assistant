# Multimodal Data Analysis & Visualization Agent

一个基于多模态内容理解的全自动数据分析可视化Agent系统。

## 功能特性

- 📊 **智能数据理解**: 支持CSV、Excel、JSON等多种数据格式
- 🖼️ **多模态输入**: 支持图表截图、表格图片的OCR识别与分析
- 🤖 **AI驱动分析**: 基于LLM自动生成分析洞察和可视化建议
- 📈 **自动可视化**: 根据数据特征自动生成最优图表类型
- 💬 **自然语言交互**: 通过对话方式探索数据和获取洞察
- 🔍 **深度分析**: 统计分析、趋势预测、异常检测等高级功能

## 技术栈

- **后端框架**: FastAPI
- **前端界面**: Streamlit
- **AI模型**: Qwen-VL (多模态), PandasAI
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly, Matplotlib, Seaborn
- **向量检索**: FAISS
- **工作流编排**: LangChain

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 启动后端API
python -m src.api_server

# 启动前端界面
streamlit run src/app.py
```

## 项目结构

```
multimodal-data-agent/
├── src/
│   ├── __init__.py
│   ├── app.py              # Streamlit前端
│   ├── api_server.py       # FastAPI后端
│   ├── agent.py            # Agent核心逻辑
│   ├── multimodal_parser.py # 多模态解析器
│   ├── data_analyzer.py    # 数据分析模块
│   ├── visualizer.py       # 可视化生成器
│   └── utils.py            # 工具函数
├── tests/
│   ├── __init__.py
│   └── test_agent.py       # 测试用例
├── data/                   # 示例数据
├── output/                 # 输出结果
├── notebooks/              # Jupyter示例
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
└── demo.py                # 演示脚本
```

## 使用示例

### Python API调用

```python
from src.agent import DataAnalysisAgent

# 初始化Agent
agent = DataAnalysisAgent()

# 上传并分析数据
result = agent.analyze("data/sales.csv")

# 自然语言查询
insight = agent.query("展示销售额趋势并预测下季度表现")

# 生成可视化报告
report = agent.generate_report(output_path="output/report.html")
```

### 多模态输入

```python
# 从图表截图提取数据并分析
result = agent.analyze_image("data/chart_screenshot.png")

# 混合文本和图片输入
result = agent.multimodal_query(
    text="分析这个销售趋势",
    image="data/sales_chart.png"
)
```

## 许可证

MIT License
