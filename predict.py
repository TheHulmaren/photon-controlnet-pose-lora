from cog import BasePredictor, Input, Path
import boto3
from boto3.session import Session
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from compel import Compel
from PIL import Image


class Predictor(BasePredictor):
  def fetch_and_save_lora(self, lora_s3_path, s3_access_key, s3_secret_key, s3_region, s3_bucket):
    session = Session(aws_access_key_id=s3_access_key,
                      aws_secret_access_key=s3_secret_key,
                      region_name=s3_region)
    s3 = session.client('s3')
    s3.download_file(s3_bucket, lora_s3_path,
                     f"./loras/{lora_s3_path.split('/')[-1]}")

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
                  default="A girl with a (black shirt)++ and (blue jeans)--",
              ),
              neg_prompt: str = Input(
                  description="The negative prompt to generate images from",
                  default="cartoon++, (low quality)++"
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
                  description="The index of the pose image to use",
                  ge=1,
                  le=12,
                  default=7,
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
              lora_s3_path: str = Input(
                  description="The Amazon S3 path of the LoRA safetensors file",
                  default=None,
              ),
              s3_access_key: str = Input(
                  description="The Amazon S3 access key",
                  default=None,
              ),
              s3_secret_key: str = Input(
                  description="The Amazon S3 secret key",
                  default=None,
              ),
              s3_region: str = Input(
                  description="The Amazon S3 region",
                  default=None,
              ),
              s3_bucket: str = Input(
                  description="The Amazon S3 bucket name",
                  default=None,
              ),
              num_of_imgs: int = Input(
                  description="The number of images to generate",
                  default=1,
              ),
              ) -> list[Path]:
    """Run a single prediction on the model"""
    self.fetch_and_save_lora(lora_s3_path, s3_access_key,
                             s3_secret_key, s3_region, s3_bucket)

    self.pipe.load_lora_weights(f"./loras/{lora_s3_path.split('/')[-1]}")

    prompt_embeds = self.compel_proc([prompt]*num_of_imgs)
    neg_prompt_embeds = self.compel_proc([neg_prompt]*num_of_imgs)
    cn_img = self.load_cn_pose_img(pose_img_index, (width, height))

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
    ).images

    output_paths = []
    for (i, image) in enumerate(images):
      output_path = f"output/out-{i}.png"
      image.save(output_path)
      output_paths.append(Path(output_path))

    return output_paths