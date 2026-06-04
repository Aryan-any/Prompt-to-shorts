from typing import List, Dict
import json
import os
import random
from pathlib import Path
import modal
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from mm_story_agent.prompts_en import role_extract_system, role_review_system, \
    story_to_image_reviser_system, story_to_image_review_system, story_to_scene_review_system,\
    story_to_scene_reviser_system, consistent_image_prompt_generator_system,\
    scene_and_image_prompt_combiner_system, consistent_image_prompt_checker_system, scene_to_image_reviser_system
from mm_story_agent.base import register_tool, init_tool_instance
from dotenv import load_dotenv
from openai import OpenAI
import requests
from PIL import Image
import io

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AttnProcessor(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def cal_attn_mask_xl(total_length,
                     id_length,
                     sa32,
                     sa64,
                     height,
                     width,
                     device="cuda",
                     dtype=torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    return mask1024, mask4096


class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self,
                 global_attn_args,
                 hidden_size=None,
                 cross_attention_dim=None,
                 id_length=4,
                 device="cuda",
                 dtype=torch.float16,
                 height=1280,
                 width=720,
                 sa32=0.5,
                 sa64=0.5,
                 ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}
        self.height = height
        self.width = width
        self.sa32 = sa32
        self.sa64 = sa64
        self.write = True

        self.global_attn_args = global_attn_args


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None
    ):
        total_count = self.global_attn_args["total_count"]
        attn_count = self.global_attn_args["attn_count"]
        cur_step = self.global_attn_args["cur_step"]
        mask1024 = self.global_attn_args["mask1024"]
        mask4096 = self.global_attn_args["mask4096"]

        if self.write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),
                                               hidden_states[:1],
                                               self.id_bank[cur_step][1].to(self.device), hidden_states[1:]))
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not self.write:
                    if hidden_states.shape[1] == (self.height // 32) * (self.width // 32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (self.height // 32) * (self.width // 32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,
                                                  :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length, 
                                                  :mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length,
                                                  self.id_length,
                                                  self.sa32,
                                                  self.sa64,
                                                  self.height,
                                                  self.width,
                                                  device=self.device, 
                                                  dtype=self.dtype)
            self.global_attn_args["mask1024"] = mask1024
            self.global_attn_args["mask4096"] = mask4096

        self.global_attn_args["attn_count"] = attn_count
        self.global_attn_args["cur_step"] = cur_step

        return hidden_states
    
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(
                -1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, sequence_length, channel).reshape(
                -1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StoryDiffusionSynthesizer:

    def __init__(self,
                 num_pages: int,
                 height: int,
                 width: int,
                 model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 id_length: int = 4,
                 num_steps: int = 50):
        self.attn_args = {
            "attn_count": 0,
            "cur_step": 0,
            "total_count": 0,
        }
        self.sa32 = 0.5
        self.sa64 = 0.5
        self.id_length = id_length
        self.total_length = num_pages
        self.height = height
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.num_steps = num_steps
        self.styles = {
            '(No style)': (
                '{prompt}',
                ''),
            'Japanese Anime': (
                'anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows',
                'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Digital/Oil Painting': (
                '{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field',
                'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Pixar/Disney Character': (
                'Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.',
                'lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo'),
            'Photographic': (
                'cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus',
                'drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Comic book': (
                'comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
                'photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Line art': (
                'line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Black and White Film Noir': (
                '{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Isometric Rooms': (
                'Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Storybook': (
                "Cartoon style, cute illustration of {prompt}.",
                'realism, photo, realistic, lowres, bad hands, bad eyes, bad arms, bad legs, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, grayscale, noisy, sloppy, messy, grainy, ultra textured'
            )
        }

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        pipe = pipe.to(self.device)
        
        # pipe.id_encoder.to(self.device)

        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(num_steps)
        unet = pipe.unet

        attn_procs = {}
        ### Insert PairedAttention
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None and (name.startswith("up_blocks") ) :
                attn_procs[name] = SpatialAttnProcessor2_0(
                    id_length=self.id_length,
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    sa32=self.sa32,
                    sa64=self.sa64,
                    global_attn_args=self.attn_args
                )
                self.attn_args["total_count"] += 1
            else:
                attn_procs[name] = AttnProcessor()
        print("successsfully load consistent self-attention")
        print(f"number of the processor : {self.attn_args['total_count']}")
        # unet.set_attn_processor(copy.deepcopy(attn_procs))
        unet.set_attn_processor(attn_procs)
        mask1024, mask4096 = cal_attn_mask_xl(
            self.total_length,
            self.id_length,
            self.sa32,
            self.sa64,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.float16,
        )

        self.attn_args.update({
            "mask1024": mask1024,
            "mask4096": mask4096
        })

        self.pipe = pipe
        self.negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation," \
                               "extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating" \
                               "limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"

    def set_attn_write(self,
                       value: bool):
        unet = self.pipe.unet
        for name, processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if cross_attention_dim is None:
                if name.startswith("up_blocks") :
                    assert isinstance(processor, SpatialAttnProcessor2_0)
                    processor.write = value

    def apply_style(self, style_name: str, positives: list, negative: str = ""):
        p, n = self.styles.get(style_name, self.styles["(No style)"])
        return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative
    
    def apply_style_positive(self, style_name: str, positive: str):
        p, n = self.styles.get(style_name, self.styles["(No style)"])
        return p.replace("{prompt}", positive) 
    
    def call(self,
             prompts: List[str],        
             input_id_images = None,
             start_merge_step = None,
             style_name: str = "Pixar/Disney Character",
             guidance_scale: float = 5.0,
             seed: int = 2047):
        assert len(prompts) == self.total_length, "The number of prompts should be equal to the number of pages."
        setup_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        torch.cuda.empty_cache()

        id_prompts = prompts[:self.id_length]
        real_prompts = prompts[self.id_length:]
        self.set_attn_write(True)
        self.attn_args.update({
            "cur_step": 0,
            "attn_count": 0
        })
        id_prompts, negative_prompt = self.apply_style(style_name, id_prompts, self.negative_prompt)
        id_images = self.pipe(
            id_prompts,
            input_id_images=input_id_images,
            start_merge_step=start_merge_step,
            num_inference_steps=self.num_steps,
            guidance_scale=guidance_scale,
            height=self.height, 
            width=self.width,
            negative_prompt=negative_prompt,
            generator=generator).images
    
        self.set_attn_write(False)
        real_images = []
        for real_prompt in real_prompts:
            self.attn_args["cur_step"] = 0
            real_prompt = self.apply_style_positive(style_name, real_prompt)
            real_images.append(self.pipe(
                real_prompt,
                num_inference_steps=self.num_steps,
                guidance_scale=guidance_scale, 
                height=self.height, 
                width=self.width,
                negative_prompt=negative_prompt,
                generator=generator).images[0]
            )

        images = id_images + real_images             
        return images


@register_tool("story_diffusion_t2i")
class StoryDiffusionAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def call(self, params: Dict):
        try:
            pages: List = params["pages"]
            save_path: str = params["save_path"]
            save_path.mkdir(exist_ok=True, parents=True)
            role_dict = self.extract_role_from_story(pages)
            # image_prompts = self.generate_image_prompt_from_story(pages)
            # print(f"\n\nImage prompts:\n {image_prompts}\n")
            scene_prompts = self.generate_scene_prompt_from_story(pages, role_dict)
            print(f"\n\nScene prompts:\n {scene_prompts}\n")
            # combined_prompts = self.combine_scene_and_image_prompt(scene_prompts, image_prompts)
            # print(f"\n\nCombined prompts:\n {combined_prompts}\n")
            image_prompts_from_scenes = self.generate_image_prompt_from_scenes(scene_prompts, role_dict)
            print(f"\n\nImage prompts from scenes:\n {image_prompts_from_scenes}\n")
            image_prompts_with_role_desc = []
            st =0
            for image_prompt in image_prompts_from_scenes:
                for role, role_desc in role_dict.items():
                    if role in image_prompt:
                        print(f"image prompts for idx{st}: {image_prompt}")
                        image_prompt = image_prompt.replace(role, role_desc)
                image_prompts_with_role_desc.append(image_prompt)
                st +=1
                # image_prompts_with_role_desc.append(combined_prompt)
            print(f"\n\nImage prompts with role desc:\n {image_prompts_with_role_desc}\n")

            images = []
            gen_id = ""
            for idx, prompt in enumerate(image_prompts_with_role_desc):
                try:
                    style = params.get("style_name", "Storybook")
                    formatted_prompt = f"realistic cartoon style, cute storybook illustration of {prompt}. High quality, detailed, vibrant colors, child-friendly and vertical portrait orientation"
                    # formatted_prompt = f"cartoon style, cute storybook illustration of {prompt}. High quality, detailed, vibrant colors, child-friendly and vertical landscape orientation, approx 9:16 aspect ratio, centered composition, optimized for vertical mobile viewing for youtube shorts, tiktok and instagram reels"
 
                    print(f"Generating image {idx +1 }/{len(image_prompts_with_role_desc)}")
                    print(f"Prompt: {formatted_prompt}")
                    
                    negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
                    max_retries = 3
                    retry_count = 0
                    error_message = ""
                    
                    while retry_count < max_retries:
                        try:
                            # response = self.client.images.generate(
                            #     model = "dall-e-3",
                            #     prompt = formatted_prompt,
                            #     size = "1024x1792",
                            #     quality = "standard",
                            #     n = 1,
                            # )
                            flux = modal.Cls.lookup("image", "FluxImage")
                            flux = flux()
                            reply = flux.fluximage.remote(formatted_prompt, negative_prompt)
                            

                            if reply:
                                # Log successful image generation
                                # llm_logger.log_call(
                                #     llm_type="dall-e",
                                #     prompt=formatted_prompt,
                                #     response=response.data[0].url,
                                #     success=True,
                                #     prompt_type="image_generation",
                                #     metadata={
                                #         "model": "dall-e-3",
                                #         "size": "1024x1792",
                                #         "quality": "standard",
                                #         "retry_count": retry_count
                                #     },
                                #     image_prompts=[formatted_prompt],
                                #     story_pages=pages
                                # )
                                break
                            else:
                                error_message = f"Empty response received on attempt {retry_count + 1}"
                                print(error_message)
                                
                        except Exception as e:
                            error_message = str(e)
                            retry_count += 1
                            if retry_count == max_retries:
                                print(f"Failed to generate image {idx + 1} after {max_retries} attempts: {str(e)}")
                                # Log failed image generation
                                # llm_logger.log_call(
                                #     llm_type="dall-e",
                                #     prompt=formatted_prompt,
                                #     response="",
                                #     success=False,
                                #     prompt_type="image_generation",
                                #     metadata={
                                #         "model": "dall-e-3",
                                #         "size": "1024x1792",
                                #         "quality": "standard",
                                #         "retry_count": retry_count
                                #     },
                                #     error_message=error_message,
                                #     image_prompts=[formatted_prompt],
                                #     story_pages=pages
                                # )
                                raise e
                            print(f"Error on attempt {retry_count}: {str(e)}. Retrying...")
                            
                    # image_url = response.data[0].url
                    # response = requests.get(image_url)
                    # image = Image.open(io.BytesIO(response.content))
                    resized_image = reply.resize((1080, 1920), Image.Resampling.LANCZOS)
                    save_file = save_path / f"p{idx +1}.png"
                    resized_image.save(save_file)
                    images.append(resized_image)
                    print(f"Successfully generated and saved image {idx +1}")
                
                except Exception as e:
                    print(f"Error generating image {idx +1}: {str(e)}")
                    continue

            return {
                "prompts": image_prompts_with_role_desc,
                "generation_results": images,
            }
        except Exception as e:
            print(f"Error in image generation process: {str(e)}")
            return {
                "error": str(e)
            }

            
    def extract_role_from_story(
            self,
            pages: List,
        ):
        print(f"\n\nExtracting roles from story:\n {len(pages)}\n")
        num_turns = self.cfg.get("num_turns", 3)
        role_extractor = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": role_extract_system.format(story_content = pages, previous_result = {}, improvement_suggestions = ""),
                "track_history": False
            }
        })
        role_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": role_review_system.format(story_content = pages, role_descriptions = {}),
                "track_history": False
            }
        })
        roles = {}
        review = ""
        for turn in range(num_turns):
            try:
                roles_response, success = role_extractor.call(role_extract_system.format(
                    story_content=pages, 
                    previous_result=roles if turn > 0 else "No previous results yet.",
                    improvement_suggestions=review
                ))
                
                print(f"Raw response from extractor: {roles_response}")
                
                # Convert response to dictionary if it's a string
                if isinstance(roles_response, str):
                    try:
                        # Try parsing as JSON first
                        roles = json.loads(roles_response.strip())
                    except json.JSONDecodeError:
                        # If that fails, try to clean up the string
                        cleaned_response = roles_response.strip()
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]
                        cleaned_response = cleaned_response.strip()
                        
                        try:
                            roles = json.loads(cleaned_response)
                        except json.JSONDecodeError:
                            # If still fails, create a simple dictionary
                            if ':' in cleaned_response:
                                roles = {}
                                lines = cleaned_response.split('\n')
                                for line in lines:
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        roles[key.strip().strip('"')] = value.strip().strip('"')
                        else:
                            # If we just have a name, create a basic entry
                            name = cleaned_response.strip('"').strip()
                            roles = {name: "Character needs description"}
                else:
                    roles = roles_response

                # Ensure we have a dictionary
                if not isinstance(roles, dict):
                    print(f"Warning: roles is not a dictionary: {type(roles)}")
                    if isinstance(roles, str):
                        roles = {roles.strip(): "Character needs description"}
                    else:
                        roles = {}

                print(f"Final roles dictionary: {roles}")
                
                # Rest of the method...
                
            except Exception as e:
                print(f"Error in extraction turn {turn + 1}: {str(e)}")
                continue
        
        # Final check to ensure we're returning a dictionary
        if not isinstance(roles, dict):
            print("Warning: Final roles is not a dictionary, returning empty dict")
            return {}
        
        # Convert roles dictionary to lowercase
        roles = {k.lower(): v.lower() for k, v in roles.items()}
        
        return roles

    def generate_image_prompt_from_story(
            self,
            pages: List,
            num_turns: int = 3
        ):
        image_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": story_to_image_review_system,
                "track_history": False
            }
        })
        image_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": story_to_image_reviser_system,
                "track_history": False
            }
        })
        image_prompts = []

        for page in pages:
            review = ""
            image_prompt = ""
            for turn in range(num_turns):
                image_prompt, success = image_prompt_reviser.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "previous_result": image_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if image_prompt.startswith("Image description:"):
                    image_prompt = image_prompt[len("Image description:"):]
                print(f"\n\nImage prompt reviser result {turn + 1}/{num_turns}:\n {image_prompt}\n")
                review, success = image_prompt_reviewer.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "image_description": image_prompt
                }, ensure_ascii=False))
                print(f"\n\nImage prompt reviewer result {turn + 1}/{num_turns}:\n {review}\n")
                if review == "Check passed.":
                    break
            image_prompts.append(image_prompt)
        return image_prompts
    
    def generate_scene_prompt_from_story(
            self,
            pages: List,
            character_descriptions: Dict,
            num_turns: int = 3
        ):
        
        scene_prompts = []
        print(f"\n\nGenerating scene prompts from story:\n {len(pages)}\n")
        scene_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": story_to_scene_review_system.format(story_content = pages,current_page = "",prev_scene_description= "", character_descriptions = character_descriptions, scene_description = ""),
                "track_history": False
            }
        })
        scene_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": story_to_scene_reviser_system.format(story_content = pages,current_page = "",prev_scene_description= "", character_descriptions = character_descriptions, previous_scene_description = "", improvement_suggestions = ""),
                "track_history": False
            }
        })

        for page in pages:
            review = ""
            scene_prompt = ""
            joined_prev = "\n\n".join(scene_prompts)
            for turn in range(num_turns):
                scene_prompt, success = scene_prompt_reviser.call(story_to_scene_reviser_system.format(story_content = pages,current_page = page,prev_scene_description= joined_prev, character_descriptions = character_descriptions, previous_scene_description = scene_prompt, improvement_suggestions = review))
                # scene_prompt, success = scene_prompt_reviser.call(json.dumps({
                #     "all_pages": pages,
                #     "current_page": page,
                #     "previous_result": scene_prompt,
                #     "All character_descriptions": character_descriptions,
                #     "improvement_suggestions": review,
                # }, ensure_ascii=False))
                if scene_prompt.startswith("Scene description:"):
                    scene_prompt = scene_prompt[len("Scene description:"):]
                print(f"\n\nScene prompt reviser result {turn + 1}/{num_turns}:\n {scene_prompt}\n")
                review, success = scene_prompt_reviewer.call(story_to_scene_review_system.format(story_content = pages,current_page = page, prev_scene_description= joined_prev, character_descriptions = character_descriptions, scene_description = scene_prompt))
                # review, success = scene_prompt_reviewer.call(json.dumps({
                #     "all_pages": pages,
                #     "current_page": page,
                #     "All character_descriptions": character_descriptions,
                #     "scene_description": scene_prompt
                # }, ensure_ascii=False))
                print(f"\n\n Scene prompt reviewer result {turn + 1}/{num_turns}:\n {review}\n")
                if review == "Check passed.":
                    break
            scene_prompts.append(scene_prompt)
        return scene_prompts
    
    def generate_image_prompt_from_scenes(
            self,
            scene_prompts: List,
            character_descriptions: Dict
        ):
        print(f"\n\nGenerating image prompts from scenes:\n {len(scene_prompts)}\n")
        print(f"Character descriptions type: {type(character_descriptions)}")
        print(f"Character descriptions content: {character_descriptions}")
        
        # Convert character_descriptions to dict if it's a string
        if isinstance(character_descriptions, str):
            try:
                character_descriptions = json.loads(character_descriptions)
            except json.JSONDecodeError:
                print("Warning: character_descriptions is a string and not valid JSON")
                character_descriptions = {}
        
        # Ensure we have a dictionary and convert to lowercase
        if not isinstance(character_descriptions, dict):
            print(f"Warning: character_descriptions is not a dictionary. Using empty dict instead.")
            character_descriptions = {}
        else:
            character_descriptions = {k.lower(): v.lower() for k, v in character_descriptions.items()}
        
        image_prompts = []
        for scene_prompt in scene_prompts:
            scene_prompt = scene_prompt.lower()  # Convert scene prompt to lowercase
            roles = {}
            for role, role_desc in character_descriptions.items():
                if role in scene_prompt:
                    roles[role] = role_desc
            
            print(f"Roles found in scene: {roles}")
            
            image_prompt_reviser = init_tool_instance({
                "tool": self.cfg.get("llm", "openai"),
                "cfg": {
                    "system_prompt": scene_to_image_reviser_system.format(
                        scene_description=scene_prompt, 
                        character_descriptions=roles
                    ),
                    "track_history": False
                }
            })
            image_prompt, success = image_prompt_reviser.call(
                scene_to_image_reviser_system.format(
                    scene_description=scene_prompt, 
                    character_descriptions=roles
                )
            )
            # Convert final image prompt to lowercase
            image_prompts.append(image_prompt.lower())
        return image_prompts
    
    def combine_scene_and_image_prompt(
            self,
            scene_prompts: List,
            image_prompts: List
        ):
        combiner = init_tool_instance({
            "tool": self.cfg.get("llm", "openai"),
            "cfg": {
                "system_prompt": scene_and_image_prompt_combiner_system,
                "track_history": False
            }
        })
        combined_prompts = []
        for scene_prompt, image_prompt in zip(scene_prompts, image_prompts):
            combined_prompt, success = combiner.call(json.dumps({
                "scene_description": scene_prompt,
                "image_description": image_prompt
            }, ensure_ascii=False))
            print(f"\n\nCombined prompt:\n {combined_prompt}\n for scene prompt: {scene_prompt}\n and image prompt: {image_prompt}\n")
            combined_prompts.append(combined_prompt)
        return combined_prompts
