import requests
import json
import os

# RunPod API 키 설정
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# 엔드포인트 ID 로드
with open("endpoints.json", "r") as f:
    endpoints = json.load(f)

# 각 엔드포인트 삭제
for gpu, endpoint_id in endpoints.items():
    response = requests.delete(
        f"https://api.runpod.io/v2/serverless/endpoints/{endpoint_id}",
        headers=headers
    )

    if response.status_code == 200:
        print(f"Successfully deleted endpoint for {gpu}: {endpoint_id}")
    else:
        print(f"Failed to delete endpoint for {gpu}: {endpoint_id}, status: {response.status_code}")

print("Cleanup completed.")