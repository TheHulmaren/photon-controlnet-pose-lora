import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


controlnet = [
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
        token="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")
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "digiplay/Photon_v1", 
    controlnet=controlnet, 
    torch_dtype=torch.float16)

pipe.save_pretrained("/src/photon-cache")
