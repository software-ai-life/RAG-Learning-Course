from collections import Counter
from pathlib import Path

from unstructured.partition.auto import partition


# PDF 檔案路徑

pdf_path = "../../data/C2/pdf/Medium.pdf"

# 使用 Unstructured 解析 PDF 內容。
elements = partition(
    filename=pdf_path,
    content_type="application/pdf",
    #strategy="fast",
)


# 顯示解析結果統計
total_characters = sum(len(str(element)) for element in elements)
print(f"總共解析出 {len(elements)} 個元素，共 {total_characters} 個字元")

# 統計不同元素類型的數量
types = Counter(element.category for element in elements)
print(f"元素類型統計: {dict(types)}")

# 顯示每個元素的內容
print("\n解析出的元素:")
for i, element in enumerate(elements, 1):
    print(f"元素 {i} ({element.category}):")
    print(element)
    print("=" * 60)
