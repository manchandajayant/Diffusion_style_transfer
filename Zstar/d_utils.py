from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableAudioPipeline
from tqdm import tqdm


class ZstarAudioPipeline(StableAudioPipeline):
    def __init__(
        self, vae, text_encoder, projection_model, tokenizer, transformer, scheduler
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            projection_model=projection_model,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        # Below is from stable audio itself
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2
        # self.inversion_scheduler = DDIMScheduler.from_config(self.scheduler.config)
        # self.inversion_scheduler.timesteps = self.inversion_scheduler.timesteps.to(self.device)
        # self.inversion_scheduler.alphas_cumprod = self.inversion_scheduler.alphas_cumprod.to(self.device)

    def next_step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ):
        """DDIM Inversion step  to add noise."""
        sched = self.inversion_scheduler
        prev_t = timestep
        next_t = (
            timestep + sched.config.num_train_timesteps // sched.num_inference_steps
        )

        alpha_t = sched.alphas_cumprod[prev_t]
        if next_t < sched.alphas_cumprod.shape[0]:
            alpha_next = sched.alphas_cumprod[next_t]
        else:
            alpha_next = torch.tensor(0.0, device=self.device, dtype=alpha_t.dtype)

        beta_t = 1 - alpha_t
        pred_x0 = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        dir_next = (1 - alpha_next).sqrt() * model_output
        return alpha_next.sqrt() * pred_x0 + dir_next, pred_x0

    def prev_step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ):
        """DDIM Denoising step to remove noise."""
        sched = self.inversion_scheduler
        prev_t = (
            timestep - sched.config.num_train_timesteps // sched.num_inference_steps
        )

        alpha_t = sched.alphas_cumprod[timestep]
        if prev_t >= 0:
            alpha_prev = sched.alphas_cumprod[prev_t]
        else:
            alpha_prev = sched.final_alpha_cumprod.to(self.device)

        beta_t = 1 - alpha_t
        pred_x0 = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        dir_prev = (1 - alpha_prev).sqrt() * model_output
        return alpha_prev.sqrt() * pred_x0 + dir_prev, pred_x0

    def get_1d_rotary_pos_embed(
        self,
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
        index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
        data type.

        Args:
            dim (`int`): Dimension of the frequency tensor.
            pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
            theta (`float`, *optional*, defaults to 10000.0):
                Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (`bool`, *optional*):
                If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            linear_factor (`float`, *optional*, defaults to 1.0):
                Scaling factor for the context extrapolation. Defaults to 1.0.
            ntk_factor (`float`, *optional*, defaults to 1.0):
                Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
            repeat_interleave_real (`bool`, *optional*, defaults to `True`):
                If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
                Otherwise, they are concateanted with themselves.
            freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
                the dtype of the frequency tensor.
        Returns:
            `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
        """
        print(type(dim), dim, "This is dimention information")
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos = torch.arange(pos)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]

        theta = theta * ntk_factor
        freqs = (
            1.0
            / (
                theta
                ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
            )
            / linear_factor
        )  # [D/2]
        freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
        is_npu = freqs.device.type == "npu"
        if is_npu:
            freqs = freqs.float()
        if use_real and repeat_interleave_real:
            # flux, hunyuan-dit, cogvideox
            freqs_cos = (
                freqs.cos()
                .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
                .float()
            )  # [S, D]
            freqs_sin = (
                freqs.sin()
                .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
                .float()
            )  # [S, D]
            return freqs_cos, freqs_sin
        elif use_real:
            # stable audio, allegro
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # complex64     # [S, D/2]
            return freqs_cis

    @torch.no_grad()
    def audio2latent(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Encode a raw waveform into latents.
        Returns float32 latents on the correct device.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.size(1) == 1:
            audio = audio.repeat(1, 2, 1)
        audio = audio.to(self.device, dtype=torch.float32)
        vae_out = self.vae.encode(audio)
        # force float
        return vae_out.latent_dist.mean.float()

    @torch.no_grad()
    def latent2audio(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back into waveform. Handle scaling symmetrically.
        """
        # undo the 0.18215 scaling
        lat = (latents / 0.18215).to(self.dtype)
        audio = self.vae.decode(lat)["sample"]
        return audio.clamp(-1, 1)

    @torch.no_grad()
    def latent2audio_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Like latent2audio but without clamping for gradient‐flow."""
        lat = (latents / 0.18215).to(self.dtype)
        return self.vae.decode(lat)["sample"]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        audio_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        latents: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        device = self.device
        if isinstance(prompt, str):
            prompt = [prompt]

        # Use the batch size from the provided latents if they exist
        batch_size = latents.shape[0] if latents is not None else len(prompt)

        # 1. Prepare Embeddings for CFG
        do_classifier_free_guidance = guidance_scale > 1.0

        # Conditional embeddings
        cond_prompt = prompt
        cond_text_embeds = self.encode_prompt(cond_prompt, device, False)
        cond_ss, cond_es = self.encode_duration(
            0.0, audio_length / self.vae.sampling_rate, device, False, batch_size
        )
        cond_encoder_states = torch.cat([cond_text_embeds, cond_ss, cond_es], dim=1)
        # cond_global_states = self.transformer.global_proj(torch.cat([cond_ss, cond_es], dim=-1))
        cond_global_states = torch.cat([cond_ss, cond_es], dim=-1)

        # Unconditional embeddings
        uncond_prompt = [""] * batch_size
        uncond_text_embeds = self.encode_prompt(uncond_prompt, device, False)
        uncond_ss, uncond_es = self.encode_duration(
            0.0, audio_length / self.vae.sampling_rate, device, False, batch_size
        )
        uncond_encoder_states = torch.cat(
            [uncond_text_embeds, uncond_ss, uncond_es], dim=1
        )
        # uncond_global_states = self.transformer.global_proj(torch.cat([uncond_ss, uncond_es], dim=-1))
        uncond_global_states = torch.cat([uncond_ss, uncond_es], dim=-1)

        # Combine for CFG
        encoder_hidden_states = torch.cat([uncond_encoder_states, cond_encoder_states])
        # global_hidden_states = torch.cat([uncond_global_states, cond_global_states])
        global_hidden_states = torch.cat([uncond_global_states, cond_global_states])

        # 2. Prepare latents
        if latents is None:
            shape = (
                batch_size,
                self.transformer.config.in_channels,
                audio_length // self.vae.hop_length,
            )
            latents = randn_tensor(shape, device=device, dtype=self.dtype)
        else:
            # Ensure provided latents are on the right device and dtype
            latents = latents.to(device=device, dtype=self.dtype)

        # 3. CRITICAL: Scale the initial latents
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Prepare rotary positional embedding
        print("This get called, self.", self.rotary_embed_dim)
        rotary_embedding = self.get_1d_rotary_pos_embed(
            32,
            latents.shape[2] + global_hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )

        # 5. Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in tqdm(self.scheduler.timesteps, desc="Generating Audio"):

            # Expand latents for CFG
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # 6. CRITICAL: Scale the model input as required by the scheduler
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.transformer(
                latent_model_input,
                t.unsqueeze(0),
                encoder_hidden_states=encoder_hidden_states,
                global_hidden_states=global_hidden_states,
                rotary_embedding=rotary_embedding,
            )[0]

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_cond)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 7. Decode to audio
        return {"sample": self.latent2audio(latents)}

    @torch.no_grad()
    def invert(
        self,
        audio: torch.Tensor,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        return_intermediates: bool = False,
        **kwargs,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        te_cond = self.encode_prompt(prompt, self.device, False)
        te_uncond = self.encode_prompt([""] * batch_size, self.device, False)
        duration_s = audio.shape[-1] / self.vae.sampling_rate
        ss, es = self.encode_duration(0.0, duration_s, self.device, False, batch_size)

        encoder_states = torch.cat(
            [
                torch.cat([te_uncond, ss, es], dim=1),
                torch.cat([te_cond, ss, es], dim=1),
            ],
            dim=0,
        )
        global_states = torch.cat(
            [
                self.transformer.global_proj(torch.cat([ss, es], dim=-1)),
                self.transformer.global_proj(torch.cat([ss, es], dim=-1)),
            ],
            dim=0,
        )

        z0 = self.audio2latent(audio) * 0.18215
        self.inversion_scheduler.set_timesteps(num_inference_steps, device=self.device)

        latents = z0.clone()
        traj = [z0]
        for t in tqdm(
            reversed(self.inversion_scheduler.timesteps), desc="Guided DDIM Inversion"
        ):
            inp = torch.cat([latents, latents], dim=0)
            noise_pred = self.transformer(
                inp,
                t.unsqueeze(0),
                encoder_hidden_states=encoder_states,
                global_hidden_states=global_states,
            )[0]
            un, co = noise_pred.chunk(2, dim=0)
            noise = un + guidance_scale * (co - un)
            latents, _ = self.next_step(noise, t, latents)
            traj.append(latents)

        zT = traj[-1]
        if return_intermediates:
            return zT, traj, z0
        return zT, z0
