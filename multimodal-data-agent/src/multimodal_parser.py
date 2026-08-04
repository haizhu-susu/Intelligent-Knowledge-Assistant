"""
多模态解析器：支持图像、表格、文档等多种格式的解析
"""

import base64
import io
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from PIL import Image
import pandas as pd

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class MultimodalParser:
    """多模态内容解析器"""
    
    def __init__(self, ocr_engine: str = "easyocr"):
        """
        初始化多模态解析器
        
        Args:
            ocr_engine: OCR引擎选择 ("easyocr" 或 "tesseract")
        """
        self.ocr_engine = ocr_engine
        self.reader = None
        
        if ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    
    def parse_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        解析图像内容，提取文本和结构化信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含提取内容的字典
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # 加载图像
        image = Image.open(image_path)
        
        result = {
            "type": "image",
            "path": str(image_path),
            "size": image.size,
            "format": image.format,
            "text_content": "",
            "structured_data": None,
            "metadata": {}
        }
        
        # OCR文本提取
        text = self._extract_text(image)
        result["text_content"] = text
        
        # 尝试检测表格结构
        if self._is_table_image(image):
            table_data = self._extract_table(image)
            result["structured_data"] = table_data
        
        return result
    
    def parse_document(self, doc_path: Union[str, Path]) -> Dict[str, Any]:
        """
        解析文档（PDF、Word等）
        
        Args:
            doc_path: 文档文件路径
            
        Returns:
            包含解析内容的字典
        """
        doc_path = Path(doc_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        result = {
            "type": "document",
            "path": str(doc_path),
            "text_content": "",
            "pages": [],
            "metadata": {}
        }
        
        # 根据扩展名处理不同类型的文档
        suffix = doc_path.suffix.lower()
        
        if suffix in ['.pdf']:
            result = self._parse_pdf(doc_path, result)
        elif suffix in ['.docx', '.doc']:
            result = self._parse_docx(doc_path, result)
        else:
            # 尝试作为文本文件读取
            with open(doc_path, 'r', encoding='utf-8') as f:
                result["text_content"] = f.read()
        
        return result
    
    def parse_tabular(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        解析表格文件（CSV、Excel等）
        
        Args:
            file_path: 表格文件路径
            
        Returns:
            pandas DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif suffix == '.json':
            df = pd.read_json(file_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return df
    
    def _extract_text(self, image: Image.Image) -> str:
        """从图像中提取文本"""
        
        if self.ocr_engine == "easyocr" and self.reader:
            results = self.reader.readtext(image)
            text = "\n".join([result[1] for result in results])
            return text
        
        elif TESSERACT_AVAILABLE:
            text = pytesseract.image_to_string(image)
            return text
        
        else:
            return "[OCR not available - install easyocr or pytesseract]"
    
    def _is_table_image(self, image: Image.Image) -> bool:
        """判断图像是否为表格"""
        # 简化的表格检测逻辑
        # 实际应用中可以使用更复杂的CV算法
        width, height = image.size
        aspect_ratio = width / height
        
        # 表格通常具有特定的宽高比和网格线
        # 这里使用启发式方法
        return 0.5 < aspect_ratio < 3.0
    
    def _extract_table(self, image: Image.Image) -> Optional[pd.DataFrame]:
        """从表格图像中提取结构化数据"""
        # 简化实现
        # 实际应用中可以使用camelot-py、tabula等库
        text = self._extract_text(image)
        
        # 尝试将文本解析为表格
        lines = text.strip().split('\n')
        if len(lines) > 1:
            # 简单的基于分隔符的解析
            data = []
            for line in lines:
                if '\t' in line:
                    data.append(line.split('\t'))
                elif '|' in line:
                    data.append([cell.strip() for cell in line.split('|') if cell.strip()])
            
            if data and len(data[0]) > 1:
                return pd.DataFrame(data[1:], columns=data[0])
        
        return None
    
    def _parse_pdf(self, pdf_path: Path, result: Dict) -> Dict:
        """解析PDF文档"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page in doc:
                text = page.get_text()
                pages_text.append(text)
            
            result["text_content"] = "\n\n".join(pages_text)
            result["pages"] = [{"page_num": i+1, "text": text} 
                             for i, text in enumerate(pages_text)]
            result["metadata"]["page_count"] = len(doc)
            
            doc.close()
            
        except ImportError:
            result["text_content"] = "[PyMuPDF not installed]"
        
        return result
    
    def _parse_docx(self, docx_path: Path, result: Dict) -> Dict:
        """解析Word文档"""
        try:
            from docx import Document
            
            doc = Document(docx_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            result["text_content"] = "\n\n".join(paragraphs)
            result["metadata"]["paragraph_count"] = len(paragraphs)
            
        except ImportError:
            result["text_content"] = "[python-docx not installed]"
        
        return result
    
    def encode_image_base64(self, image_path: Union[str, Path]) -> str:
        """将图像编码为base64字符串"""
        image_path = Path(image_path)
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """获取图像基本信息"""
        image_path = Path(image_path)
        image = Image.open(image_path)
        
        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "width": image.width,
            "height": image.height
        }
