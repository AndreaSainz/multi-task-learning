from code.models.score_unet import ContextUnet, ddpm_schedules
import torch

ckpt = torch.load("/home/as3628/multi-task-learning/simpler_pipeline/model_39.pth", map_location="cuda")
print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())