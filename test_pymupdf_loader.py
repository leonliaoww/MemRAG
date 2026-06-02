"""测试 PyMuPDFLoader 的基本功能（不启用 OCR）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.engine.ingestion.loader import PyMuPDFLoader


def test_pymupdf_loader():
    pdf_path = r"C:\Users\LL\Desktop\文字文稿1.pdf"

    print(f"正在测试 PyMuPDFLoader.load() (OCR 禁用): {pdf_path}")
    print("-" * 60)

    # 检查文件是否存在
    if not Path(pdf_path).exists():
        print(f"错误: 文件不存在 - {pdf_path}")
        return

    try:
        # 创建 loader 实例，禁用 OCR
        loader = PyMuPDFLoader(pdf_path, ocr_enabled=False)
        
        # 调用 load 方法
        docs = loader.load()

        print(f"PyMuPDFLoader.load() 返回: {len(docs)} 个文档")
        print("")
        
        if len(docs) == 0:
            print("提示: 文档返回为空，可能是扫描件（图像格式）")
            print("需要安装 Tesseract OCR 才能处理扫描件")
            print("")
            print("安装步骤:")
            print("1. 下载: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. 安装时勾选中文语言包")
            print("3. 将 tesseract.exe 路径添加到系统 PATH")
        else:
            for i, doc in enumerate(docs[:5]):
                print(f"\n文档 {i+1}:")
                print(f"  页码: {doc.metadata.get('page')}")
                print(f"  来源类型: {doc.metadata.get('source_type')}")
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"  内容:\n{content_preview}")

        print("\n" + "=" * 60)
        print("测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pymupdf_loader()
