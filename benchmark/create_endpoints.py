import requests
import json
import os

# RunPod API 키 설정
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# 테스트할 GPU 목록
gpu_types = [
    "NVIDIA RTX A6000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A4000",
    "NVIDIA L4"
]

endpoints = {}

# 각 GPU 유형에 대해 엔드포인트 생성
for gpu in gpu_types:
    data = {
        "name": f"bge-m3-ko-{gpu.replace(' ', '-').lower()}",
        "image": "ghcr.io/jclab-joseph/runpod-bge-m3-ko-serverless:latest",
        "gpuCount": 1,
        "gpuType": gpu,
        "containerDiskSizeGB": 10,
        "minWorkers": 1,
        "maxWorkers": 5,
        "idleTimeout": 60,  # 1분
        "networkVolumeSize": 30
    }

    response = requests.post(
        "https://api.runpod.io/v2/serverless/endpoints",
        headers=headers,
        data=json.dumps(data)
    )

    result = response.json()
    if response.status_code == 200:
        endpoints[gpu] = result["id"]
        print(f"Created endpoint for {gpu}: {result['id']}")
    else:
        print(f"Failed to create endpoint for {gpu}: {result}")

# 엔드포인트 ID를 파일로 저장
with open("endpoints.json", "w") as f:
    json.dump(endpoints, f, indent=2)

print("All endpoints created and saved to endpoints.json")