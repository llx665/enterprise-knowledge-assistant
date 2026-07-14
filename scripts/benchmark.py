# ============================================================
# 简易压测脚本
# 测试单机 8G 内存环境下的 QPS 与延迟分布
# ============================================================
import asyncio
import time
import statistics
import json
from typing import List

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "aiohttp"])
    import aiohttp

BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 50
CONCURRENCY = 5

SAMPLE_QUERIES = [
    "Python 中如何定义函数？",
    "什么是列表推导式？",
    "请解释 Python 装饰器",
    "企业考勤制度有哪些？",
    "如何安装 Python 包？",
    "什么是上下文管理器？",
    "Python 文件操作",
    "异常处理的方法",
    "类和对象的区别",
    "多线程编程",
]


async def single_request(session, query, idx) -> dict:
    """执行一次问答请求"""
    start = time.time()
    try:
        async with session.post(
            f"{BASE_URL}/api/chat/ask",
            json={"query": query, "session_id": f"bench_{idx}", "top_k": 3},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            # 读取完整 SSE 响应
            full_text = await resp.text()
            elapsed = (time.time() - start) * 1000
            lines = full_text.strip().split("\n")
            token_count = sum(1 for l in lines if '"token"' in l)
            return {
                "success": resp.status == 200,
                "status": resp.status,
                "elapsed_ms": round(elapsed, 1),
                "token_count": token_count,
                "error": None,
            }
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {
            "success": False,
            "status": 0,
            "elapsed_ms": round(elapsed, 1),
            "token_count": 0,
            "error": str(e),
        }


async def run_benchmark():
    """运行压测"""
    print("=" * 60)
    print("  企业智能知识库助手 - 压测脚本")
    print(f"  目标: {BASE_URL}")
    print(f"  请求数: {NUM_REQUESTS}, 并发: {CONCURRENCY}")
    print("=" * 60)

    # 检查服务健康状态
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=5) as resp:
                if resp.status != 200:
                    print(f"❌ 服务不可用 (HTTP {resp.status})")
                    return
                print("✅ 服务健康检查通过")
    except Exception as e:
        print(f"❌ 服务连接失败: {e}")
        return

    # 执行压测
    all_results: List[dict] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def limited_request(q, idx):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await single_request(session, q, idx)

    tasks = []
    for i in range(NUM_REQUESTS):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        tasks.append(limited_request(query, i))

    print(f"\n⏳ 执行 {NUM_REQUESTS} 个请求...")
    start_total = time.time()
    all_results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - start_total

    # 统计结果
    success_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    latencies = [r["elapsed_ms"] for r in success_results]

    print("\n" + "=" * 60)
    print("  📊 压测结果")
    print("=" * 60)
    print(f"  总请求数:     {NUM_REQUESTS}")
    print(f"  成功数:       {len(success_results)}")
    print(f"  失败数:       {len(failed_results)}")
    print(f"  成功率:       {len(success_results)/NUM_REQUESTS*100:.1f}%")
    print(f"  总耗时:       {total_elapsed:.2f}s")

    if latencies:
        print(f"\n  ⚡ 延迟分布 (ms):")
        print(f"    最小:    {min(latencies):.1f} ms")
        print(f"    最大:    {max(latencies):.1f} ms")
        print(f"    平均:    {statistics.mean(latencies):.1f} ms")
        if len(latencies) > 1:
            print(f"    中位数:  {statistics.median(latencies):.1f} ms")
            print(f"    P95:     {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")
            print(f"    P99:     {sorted(latencies)[int(len(latencies)*0.99)]:.1f} ms")
        print(f"    QPS:     {len(success_results)/total_elapsed:.1f}")

    if failed_results:
        print(f"\n  ❌ 失败详情:")
        for r in failed_results[:5]:
            print(f"    - {r['error']}")

    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"base_url": BASE_URL, "num_requests": NUM_REQUESTS, "concurrency": CONCURRENCY},
        "results": {
            "total": NUM_REQUESTS,
            "success": len(success_results),
            "failed": len(failed_results),
            "success_rate": round(len(success_results)/NUM_REQUESTS*100, 1),
            "total_time_s": round(total_elapsed, 2),
            "qps": round(len(success_results)/total_elapsed, 1) if total_elapsed > 0 else 0,
        },
        "latency_ms": {
            "min": round(min(latencies), 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
            "avg": round(statistics.mean(latencies), 1) if latencies else 0,
            "p95": round(sorted(latencies)[int(len(latencies)*0.95)], 1) if len(latencies) > 20 else 0,
            "p99": round(sorted(latencies)[int(len(latencies)*0.99)], 1) if len(latencies) > 50 else 0,
        },
    }

    report_path = f"./data/logs/benchmark_{int(time.time())}.json"
    import os, json as j
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        j.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📝 报告已保存: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
