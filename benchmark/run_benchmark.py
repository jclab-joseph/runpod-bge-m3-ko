import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

# RunPod API 키 설정
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# 엔드포인트 ID 로드
with open("endpoints.json", "r") as f:
    endpoints = json.load(f)

# 벤치마크를 위한 텍스트 데이터 준비
# 다양한 길이의 텍스트를 준비
short_text = ["안녕하세요. 반갑습니다." for _ in range(10)]
medium_text = ["이 문장은 임베딩 벤치마크를 위한 중간 길이의 텍스트입니다. 한국어 모델의 성능을 테스트하기 위해 작성되었습니다." for _ in range(10)]
long_text = [
    "인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현하려는 컴퓨터 과학의 세부분야 중 하나이다. 컴퓨터가 인간의 지능적인 행동을 모방할 수 있도록 하는 방법을 연구하는 분야로서, 언어의 이해나 학습 등 인간이 가진 지적 능력을 컴퓨터에서 구현하는 기술이다. 인공지능은 기계학습, 자연어 처리, 로보틱스, 컴퓨터 비전 등의 분야를 포함하며, 최근에는 딥러닝 기술의 발전으로 큰 주목을 받고 있다."
    for _ in range(10)]


# 벤치마크 함수 정의
def run_benchmark(endpoint_id, texts, batch_size, name):
    url = f"https://api.runpod.io/v2/{endpoint_id}/run"

    data = {
        "input": {
            "texts": texts,
            "batch_size": batch_size,
            "normalize_embeddings": True
        }
    }

    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    request_time = time.time() - start_time

    if response.status_code != 200:
        print(f"Error in request: {response.text}")
        return {
            "name": name,
            "batch_size": batch_size,
            "text_count": len(texts),
            "success": False,
            "request_time": request_time,
            "total_time": None,
            "error": response.text
        }

    result = response.json()

    # 작업 완료까지 대기
    status_url = f"https://api.runpod.io/v2/{endpoint_id}/status/{result['id']}"
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()

        if status_data["status"] == "COMPLETED":
            break
        elif status_data["status"] in ["FAILED", "CANCELLED"]:
            return {
                "name": name,
                "batch_size": batch_size,
                "text_count": len(texts),
                "success": False,
                "request_time": request_time,
                "total_time": time.time() - start_time,
                "error": status_data.get("output", "Unknown error")
            }

        time.sleep(0.5)

    total_time = time.time() - start_time

    return {
        "name": name,
        "batch_size": batch_size,
        "text_count": len(texts),
        "success": True,
        "request_time": request_time,
        "total_time": total_time,
        "throughput": len(texts) / total_time
    }


# 벤치마크 실행
results = []

# 다양한 배치 사이즈와 텍스트 길이로 테스트
batch_sizes = [1, 5, 10]
text_datasets = [
    {"name": "short_text", "data": short_text},
    {"name": "medium_text", "data": medium_text},
    {"name": "long_text", "data": long_text}
]

for gpu_name, endpoint_id in endpoints.items():
    for batch_size in batch_sizes:
        for dataset in text_datasets:
            # 워밍업 실행 (결과 무시)
            run_benchmark(endpoint_id, dataset["data"][:5], batch_size, f"{gpu_name}-{dataset['name']}")

            # 실제 벤치마크 (3번 실행하여 평균)
            for i in range(3):
                result = run_benchmark(endpoint_id, dataset["data"], batch_size, f"{gpu_name}-{dataset['name']}")
                result["run"] = i + 1
                result["gpu"] = gpu_name
                result["text_type"] = dataset["name"]
                results.append(result)
                print(f"Completed: {gpu_name}, {dataset['name']}, batch_size={batch_size}, run={i + 1}")

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(results)

# 성공한 결과만 필터링
successful_df = df[df["success"] == True]

# 평균 계산
avg_results = successful_df.groupby(["gpu", "text_type", "batch_size"]).agg({
    "total_time": "mean",
    "throughput": "mean"
}).reset_index()

# CSV로 저장
df.to_csv("benchmark_results_raw.csv", index=False)
avg_results.to_csv("benchmark_results_avg.csv", index=False)

# 그래프 생성
plt.figure(figsize=(15, 10))
for i, text_type in enumerate(["short_text", "medium_text", "long_text"]):
    plt.subplot(3, 1, i + 1)
    data = avg_results[avg_results["text_type"] == text_type]

    # 각 GPU 유형별로 그룹화
    for gpu in endpoints.keys():
        gpu_data = data[data["gpu"] == gpu]
        plt.plot(gpu_data["batch_size"], gpu_data["throughput"], marker='o', label=gpu)

    plt.title(f"Throughput for {text_type}")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (texts/sec)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("benchmark_results.png")
plt.close()

# 결과 출력
print("\nBenchmark Results Summary:")
for gpu in endpoints.keys():
    print(f"\n{gpu}:")
    for text_type in ["short_text", "medium_text", "long_text"]:
        gpu_text_data = avg_results[(avg_results["gpu"] == gpu) & (avg_results["text_type"] == text_type)]
        if not gpu_text_data.empty:
            max_throughput = gpu_text_data["throughput"].max()
            best_batch = gpu_text_data.loc[gpu_text_data["throughput"].idxmax()]["batch_size"]
            print(f"  {text_type}: Best throughput {max_throughput:.2f} texts/sec at batch size {best_batch}")

print("\nComplete benchmark results saved to benchmark_results_avg.csv")
print("Raw data saved to benchmark_results_raw.csv")
print("Graphs saved to benchmark_results.png")