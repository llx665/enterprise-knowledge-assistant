# ============================================================
# 文档解析单元测试
# ============================================================
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.parser import parser, DocumentParser


def test_parse_markdown():
    """测试 Markdown 文档解析"""
    content = """# Test Document

## Section 1

This is the first paragraph with some content.

## Section 2

This is the second paragraph.
"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        fpath = f.name

    try:
        result = parser.parse(fpath, "test.md")
        assert len(result) > 0, "Should produce at least one parent chunk"
        total_chunks = sum(len(p["chunks"]) for p in result)
        assert total_chunks > 0, "Should produce child chunks"
        print(f"  PASS: Markdown parsing -> {len(result)} parent chunks, {total_chunks} child chunks")
    finally:
        os.unlink(fpath)


def test_parse_empty():
    """测试空文档解析"""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("")
        fpath = f.name

    try:
        result = parser.parse(fpath, "empty.txt")
        assert len(result) == 0, "Empty document should produce no chunks"
        print("  PASS: Empty document parsing")
    finally:
        os.unlink(fpath)


def test_clean_text():
    """测试文本清洗"""
    dirty = "Hello\x00World\r\n\r\n\r\n  Extra   spaces  "
    clean = parser._clean_text(dirty)
    assert "\x00" not in clean, "Null chars should be removed"
    assert "\r" not in clean, "CR should be normalized"
    print(f"  PASS: Text cleaning -> '{clean}'")


def test_supported_extensions():
    """测试支持的文件格式"""
    assert DocumentParser().SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}
    print("  PASS: Supported extensions")


if __name__ == "__main__":
    print("Running parser tests...")
    test_parse_markdown()
    test_parse_empty()
    test_clean_text()
    test_supported_extensions()
    print("All parser tests passed!")
