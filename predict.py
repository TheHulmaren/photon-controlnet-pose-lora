from pathlib import Path
import os
import os.path
from cog import BasePredictor, Input, Path
import requests
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from compel import Compel
from PIL import Image


class Predictor(BasePredictor):
  def fetch_and_save_lora(self, lora_url):
    # Check if loras folder exists and the file is already there
    Path.mkdir(Path("./loras"), exist_ok=True)

    loras_cnt = len([name for name in os.listdir('./loras')
                    if os.path.isfile(os.path.join('./loras', name))])
    dest = f"./loras/lora-{loras_cnt+1}.safetensors"
    resp = requests.get(lora_url)

    with open(dest, 'wb') as f:
      f.write(resp.content)

    return dest

  def load_cn_pose_img(self, index=7, resize=(512, 768)):
    if (index < 1 or index > 12):
      raise ValueError("Pose image index must be between 1 and 12")
    return [Image.open(f"./poses/Closeup ({index}).png").convert("RGB").resize(resize)]

  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    controlnet = [
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
            token="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")
    ]

    self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "digiplay/Photon_v1", controlnet=controlnet, torch_dtype=torch.float16)
    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        self.pipe.scheduler.config, use_karras_sigmas=True)

    self.compel_proc = Compel(tokenizer=self.pipe.tokenizer,
                              text_encoder=self.pipe.text_encoder)

    self.pipe = self.pipe.to("cuda")

  # The arguments and types the model takes as input
  def predict(self,
              prompt: str = Input(
                  description="The prompt to generate images from",
                  default="realistic, hyperrealism, best quality, masterpiece, ultra high res, photorealistic++++, (beautiful eyes)++, (soft lighting), portrait, (upper body), a wonderful photo of girl, wearing gorgeous dress on high class party",
              ),
              neg_prompt: str = Input(
                  description="The negative prompt to generate images from",
                  default="(worst quality)++++, (low quality)++++, (normal quality)++, (bad skin)++, disfigured, cartoon, painting, illustration, nsfw"
              ),
              num_of_imgs: int = Input(
                  description="The number of images to generate",
                  default=1,
              ),
              width: int = Input(
                  description="The width of the generated image",
                  default=512,
              ),
              height: int = Input(
                  description="The height of the generated image",
                  default=768,
              ),
              pose_img_index: int = Input(
                  description="The index of the pose image to use. More info: https://civitai.com/models/256902/openpose-portrait-poses",
                  ge=1,
                  le=12,
                  default=8,
              ),
              cn_guidance_strength: float = Input(
                  description="The strength of the pose guidance",
                  ge=0.0,
                  le=1.0,
                  default=0.25,
              ),
              cn_guidance_start: float = Input(
                  description="The start of the pose guidance",
                  ge=0.0,
                  le=1.0,
                  default=0.0,
              ),
              cn_guidance_end: float = Input(
                  description="The end of the pose guidance",
                  ge=0.0,
                  le=1.0,
                  default=0.25,
              ),
              loras_url: str = Input(
                  description="List of LoRA safetensors url and weights formatted as: [url/to/lora1.safetensors|0.6], [url/to/lora2.safetensors|0.4]",
                  default=None,
              )
              ) -> list[Path]:
    """Run a single prediction on the model"""

    loras_url = loras_url.replace(" ", "")
    lora_urls = [l.split('|')[0][1:] for l in loras_url.split(",")]
    lora_weights = [float(l.split('|')[1][:-1]) for l in loras_url.split(",")]

    print(len(lora_urls), 'LoRA URLs:', lora_urls)
    print(len(lora_weights), 'LoRA weights:', lora_weights)

    # fetch LoRAs from s3
    lora_paths = []
    print('Fetching LoRAs from URLs...')
    for lora in lora_urls:
      lora_paths.append(self.fetch_and_save_lora(lora))

    # set LoRA weights
    print('Setting LoRA weights...')
    for i, lora_path in enumerate(lora_paths):
      self.pipe.load_lora_weights(
          lora_path, adapter_name=str(i))
    self.pipe.set_adapters([str(i) for i in range(
        len(lora_paths))], adapter_weights=lora_weights)

    # set prompt and negative prompt embeddings
    prompt_embeds = self.compel_proc([prompt]*num_of_imgs)
    neg_prompt_embeds = self.compel_proc([neg_prompt]*num_of_imgs)

    # load pose image
    print('Loading pose image...')
    cn_img = self.load_cn_pose_img(pose_img_index, (width, height))

    # generate images
    print('Generating images...')
    images = self.pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_prompt_embeds,
        image=cn_img,
        num_inference_steps=30,
        width=width,
        height=height,
        guidance_scale=7.0,
        controlnet_conditioning_scale=cn_guidance_strength,
        control_guidance_start=cn_guidance_start,
        control_guidance_end=cn_guidance_end,
        cross_attention_kwargs={"scale": 1},
    ).images

    # unload LoRAs
    # if not unloaded, the next prediction will use the same LoRAs
    self.pipe.unload_lora_weights()

    Path.mkdir(Path("./output"), exist_ok=True)
    output_paths = []
    for (i, image) in enumerate(images):
      output_path = f"./output/out-{i}.png"
      image.save(output_path)
      output_paths.append(Path(output_path))

    print('Images generated successfully!')
    return output_paths
