"""Visual Prompt Tuning (VPT) wrapper for ViT-B/16.

I implement shallow VPT following:
    Jia et al. (2022), "Visual Prompt Tuning"
    https://arxiv.org/abs/2203.12119

Learnable prompt tokens are prepended to the patch embedding sequence
immediately after the CLS token. Because transformer self-attention is
global, the encoder handles the extended sequence with no architectural
changes to the ViT blocks. The CLS token stays at index 0 and is still
used for classification.

Why "shallow" vs "deep":
    Shallow VPT inserts prompts only at the input (before the first
    transformer layer). Deep VPT inserts a fresh set at every layer.
    Shallow is simpler to implement, adds fewer parameters, and is a
    reasonable starting point for comparison.

How this fits the ablation study:
    Condition 4 (full fine-tuning) + num_prompts > 0 gives you a variant
    that combines unrestricted weight updates with explicit learned input
    tokens. This directly tests the group notes' claim that prompt tokens
    "can over-influence the model" when attention is already unconstrained.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class PromptTunedViT(nn.Module):
    """I wrap ViTForImageClassification with learnable shallow prompt tokens.

    Args:
        vit_model: A ViTForImageClassification instance with any freeze
            configuration already applied.
        num_prompts: Number of learnable tokens to insert. 10-50 is typical
            in the VPT paper; more tokens = more capacity but more risk of
            over-influencing attention.
        prompt_dropout: Dropout rate on prompt embeddings during training.
    """

    def __init__(
        self,
        vit_model: ViTForImageClassification,
        num_prompts: int = 10,
        prompt_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vit_model = vit_model
        self.num_prompts = num_prompts
        hidden_size = vit_model.config.hidden_size  # 768 for ViT-B/16

        # I initialize prompts with small values so early training is stable
        # and the model starts close to the non-prompted baseline.
        self.prompt_embeddings = nn.Parameter(
            torch.empty(1, num_prompts, hidden_size)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """I run the prompted forward pass using a forward hook.

        I register a one-shot hook on the embedding layer to inject prompt
        tokens after the standard patch+CLS+position embeddings are computed.
        The full model forward pass then runs unmodified, so this approach
        is robust to any internal API changes in transformers.
        """
        batch_size = pixel_values.shape[0]
        prompts = self.prompt_dropout(
            self.prompt_embeddings.expand(batch_size, -1, -1)
        )

        # Hook intercepts embeddings output and inserts prompt tokens.
        # Layout after injection: [CLS | p1...pN | patch1...patch196]
        def _inject(module, input, output):
            return torch.cat([
                output[:, :1, :],   # CLS token  (B, 1, 768)
                prompts,            # prompt tokens (B, N, 768)
                output[:, 1:, :],   # 196 patch tokens (B, 196, 768)
            ], dim=1)

        handle = self.vit_model.vit.embeddings.register_forward_hook(_inject)
        try:
            outputs = self.vit_model(pixel_values, labels=labels)
        finally:
            handle.remove()

        return outputs

    def count_prompt_parameters(self) -> int:
        """I return the number of trainable prompt parameters."""
        return self.prompt_embeddings.numel()
