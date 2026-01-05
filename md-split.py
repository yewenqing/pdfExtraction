from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
import json

with open("document.md") as f:
    markdown_document =  f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)




def save_splits_to_disk(splits, base_dir="document_slices"):
    if os.path.exists(base_dir):
        os.rmdir(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for i, doc in enumerate(splits):
        # 从 metadata 中提取标题，构建文件名（过滤掉非法字符）
        h1 = doc.metadata.get("Header 1", "无标题").replace("/", "_")
        h2 = doc.metadata.get("Header 2", "").replace("/", "_")
        h3 = doc.metadata.get("Header 3", "").replace("/", "_")
        
        filename = f"{i:03d}_{h1}_{h2}_{h3}.txt"
        file_path = os.path.join(base_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            # 在文件开头写入元数据，方便查看
            f.write(f"METADATA: {str(doc.metadata)}\n")
            f.write("-" * 20 + "\n")
            f.write(doc.page_content)





def save_to_jsonl(splits, filename="splits.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for doc in splits:
            # 将 Document 对象转为字典
            data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

# 保存
save_to_jsonl(md_header_splits)
save_splits_to_disk(md_header_splits)


# --- 以后如何读取回来 ---
def load_from_jsonl(filename="splits.jsonl"):
    from langchain_core.documents import Document
    new_splits = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            new_splits.append(Document(**data))
    return new_splits

print("mdrun")