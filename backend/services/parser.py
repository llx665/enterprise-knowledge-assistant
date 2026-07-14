# ============================================================
# 文档解析模块
# 支持 PDF / Word / HTML / Markdown 多格式解析
# 父子分层分块策略 + 文本清洗去噪
# ============================================================
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from backend.config import get_settings
from backend.models.schemas import ChunkInfo


class DocumentParser:
    """多格式文档解析器，负责文件解析、分块与清洗"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}

    def __init__(self):
        self.settings = get_settings()

    def parse(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        解析文档，返回父子分块结果列表
        每个元素: {"parent_id": str, "parent_content": str, "chunks": [ChunkInfo, ...]}
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_EXTENSIONS}")

        # 读取原始文本
        raw_text, metadata = self._extract_text(file_path, filename, ext)

        # 文本清洗
        clean_text = self._clean_text(raw_text)

        # 父子分层分块
        parent_chunks = self._parent_child_chunk(clean_text, metadata)

        logger.info(f"文档解析完成: {filename} | {len(clean_text)} 字符 | {len(parent_chunks)} 父块")
        return parent_chunks

    def _extract_text(self, file_path: str, filename: str, ext: str) -> Tuple[str, Dict]:
        """根据文件类型提取文本"""
        metadata = {"filename": filename, "file_type": ext, "source_path": file_path}

        if ext == ".pdf":
            return self._extract_pdf(file_path), metadata
        elif ext == ".docx":
            return self._extract_docx(file_path), metadata
        elif ext in (".html", ".htm"):
            return self._extract_html(file_path), metadata
        elif ext == ".md":
            return self._extract_markdown(file_path), metadata
        elif ext == ".txt":
            return self._extract_txt(file_path), metadata
        return "", metadata

    # ---------- 各格式提取方法 ----------

    def _extract_pdf(self, file_path: str) -> str:
        """提取 PDF 文本（含页码标注）"""
        from pypdf import PdfReader
        texts = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    texts.append(f"[page_{i + 1}]\n{text}")
        except Exception as e:
            logger.error(f"PDF 解析失败 {file_path}: {e}")
            # 降级返回文件名
            texts.append(f"[PDF 解析错误: {e}]")
        return "\n\n".join(texts)

    def _extract_docx(self, file_path: str) -> str:
        """提取 Word 文本"""
        from docx import Document
        texts = []
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    texts.append(para.text)
        except Exception as e:
            logger.error(f"DOCX 解析失败 {file_path}: {e}")
            texts.append(f"[DOCX 解析错误: {e}]")
        return "\n\n".join(texts)

    def _extract_html(self, file_path: str) -> str:
        """提取 HTML 文本"""
        from bs4 import BeautifulSoup
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")
            # 移除脚本和样式
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text
        except Exception as e:
            logger.error(f"HTML 解析失败 {file_path}: {e}")
            return ""

    def _extract_markdown(self, file_path: str) -> str:
        """提取 Markdown 文本"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Markdown 解析失败 {file_path}: {e}")
            return ""

    def _extract_txt(self, file_path: str) -> str:
        """提取纯文本"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT 解析失败 {file_path}: {e}")
            return ""

    # ---------- 文本清洗 ----------

    def _clean_text(self, text: str) -> str:
        """文本去噪、标准化"""
        if not text:
            return ""

        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 移除非打印字符（保留中文、英文、数字、标点）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # 合并连续空白行（保留最多一个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 去除多余空格（保留行内单空格）
        text = re.sub(r" +", " ", text)

        # 去除 URL（可选保留）
        text = re.sub(r"https?://\S+", "[链接]", text)

        # 去除冗余分隔线
        text = re.sub(r"[-*=]{4,}", "", text)

        return text.strip()

    # ---------- 父子分层分块 ----------

    def _parent_child_chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """
        父子分层分块策略：
        - 父块：按段落实体分割（如章节标题），大小为 parent_chunk_size
        - 子块：从父块中按 chunk_size + overlap 切分
        - 每个子块附带父块 ID，方便溯源
        """
        if not text.strip():
            return []

        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap
        parent_size = self.settings.parent_chunk_size

        # 1) 初步按标题/空行分割段落作为候选父块
        paragraphs = self._split_into_paragraphs(text)

        # 2) 合并小段落形成父块
        parent_blocks = self._merge_paragraphs(paragraphs, parent_size)

        # 3) 对每个父块做子块切分
        result = []
        for parent_idx, parent_text in enumerate(parent_blocks):
            parent_id = f"parent_{metadata.get('filename', 'doc')}_{parent_idx}"
            child_chunks = self._split_into_chunks(parent_text, chunk_size, chunk_overlap)

            chunks = []
            for child_idx, child_text in enumerate(child_chunks):
                if not child_text.strip():
                    continue
                # 尝试提取页码信息
                page_match = re.search(r"\[page_(\d+)\]", child_text)
                page_num = int(page_match.group(1)) if page_match else None

                chunk = ChunkInfo(
                    chunk_id=f"{parent_id}_chunk_{child_idx}",
                    content=child_text.strip(),
                    parent_id=parent_id,
                    page_number=page_num,
                    paragraph_number=child_idx + 1,
                    metadata={
                        "doc_id": metadata.get("filename", ""),
                        "filename": metadata.get("filename", ""),
                        "file_type": metadata.get("file_type", ""),
                        "parent_idx": parent_idx,
                        "child_idx": child_idx,
                    },
                )
                chunks.append(chunk)

            if chunks:
                result.append({
                    "parent_id": parent_id,
                    "parent_content": parent_text[:200],  # 摘要存父块开头
                    "chunks": chunks,
                })

        return result

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按标题或空行分割为段落"""
        # 尝试按 Markdown 标题分割
        lines = text.split("\n")
        paragraphs = []
        current = []

        for line in lines:
            # 标题行作为新段落开始
            if re.match(r"^#{1,4}\s", line) or re.match(r"^[A-Z][^。！？\n]{0,20}[：:]?\s*$", line):
                if current:
                    paragraphs.append("\n".join(current))
                current = [line]
            elif line.strip() == "":
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
            else:
                current.append(line)

        if current:
            paragraphs.append("\n".join(current))

        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs(self, paragraphs: List[str], max_size: int) -> List[str]:
        """合并小段落为父块"""
        merged = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < max_size:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    merged.append(current)
                current = para

        if current:
            merged.append(current)

        return merged

    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """按 token 数（字符近似）滑动窗口切分子块"""
        # 简单按字符切分（生产环境可替换为 tokenizer）
        chunks = []
        start = 0
        text_len = len(text)

        if text_len <= chunk_size:
            return [text]

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)

            # 移动窗口（带 overlap）
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks


# 全局解析器实例
parser = DocumentParser()
