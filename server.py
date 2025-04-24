import runpod
from handler import ModelHandler

model = ModelHandler()

def handler(job):
    return model(job)

runpod.serverless.start({"handler": handler})
