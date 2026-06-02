import requests
import time

BASE_URL = "https://mineru.net/api/v1/agent"


def parse_by_file(file_path, language="ch", page_range=None, enable_table=True, is_ocr=False, enable_formula=True):
    """通过文件上传提交文档解析任务并等待结果。"""
    file_name = file_path.split("/")[-1].split("\\")[-1]

    # 1. 获取签名上传 URL
    data = {"file_name": file_name, "language": language, "enable_table": enable_table, "is_ocr": is_ocr, "enable_formula": enable_formula}
    if page_range:
        data["page_range"] = page_range

    resp = requests.post(f"{BASE_URL}/parse/file", json=data)
    result = resp.json()
    if result["code"] != 0:
        print(f"获取上传链接失败: {result['msg']}")
        return None

    task_id = result["data"]["task_id"]
    file_url = result["data"]["file_url"]
    print(f"任务已创建, task_id: {task_id}")

    # 2. PUT 上传文件到 OSS
    with open(file_path, "rb") as f:
        put_resp = requests.put(file_url, data=f)
        if put_resp.status_code not in (200, 201):
            print(f"文件上传失败, HTTP {put_resp.status_code}")
            return None
    print("文件上传成功，等待解析...")

    # 3. 轮询等待结果
    return poll_result(task_id)


def poll_result(task_id, timeout=300, interval=3):
    """轮询查询解析结果。"""
    state_labels = {
        "pending": "排队中",
        "running": "解析中",
        "waiting-file": "等待文件上传",
    }
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/parse/{task_id}")
        result = resp.json()
        state = result["data"]["state"]
        elapsed = int(time.time() - start)

        if state == "done":
            markdown_url = result["data"]["markdown_url"]
            print(f"[{elapsed}s] 解析完成, Markdown 下载链接: {markdown_url}")
            
            # 下载并返回 Markdown 内容
            md_resp = requests.get(markdown_url)
            return {
                "task_id": task_id,
                "markdown_url": markdown_url,
                "content": md_resp.text
            }

        if state == "failed":
            print(f"[{elapsed}s] 解析失败: {result['data'].get('err_msg', '未知错误')}")
            return None

        print(f"[{elapsed}s] {state_labels.get(state, state)}...")
        time.sleep(interval)

    print(f"轮询超时 ({timeout}s)，请稍后手动查询 task_id: {task_id}")
    return None


if __name__ == "__main__":
    # 解析 PDF 文件
    result = parse_by_file(r"C:\Users\LL\Desktop\93a87358df58e5a7ccfe970b7d63ab1e.jpg")
    
    if result:
        print("\n" + "=" * 60)
        print("解析结果:")
        print("=" * 60)
        
        # 1. 保存到本地文件
        output_file = r"C:\Users\LL\Desktop\文字文稿1.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["content"])
        print(f"已保存到: {output_file}")
        
        # 2. 打印内容预览
        print("\n内容预览:")
        print("-" * 60)
        content_preview = result["content"][:1000] + "..." if len(result["content"]) > 1000 else result["content"]
        print(content_preview)
