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
from transformers.modeling_outputs import ImageClassifierOutput


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
    ) -> ImageClassifierOutput:
        """I run the prompted forward pass.

        I intercept between the ViT embedding layer and the encoder to
        inject prompt tokens. The rest of the forward pass is standard.
        """
        batch_size = pixel_values.shape[0]
        vit = self.vit_model.vit

        # Standard patch + CLS + position embeddings
        # Output shape: (B, 1 + 196, 768) = (B, 197, 768)
        embedding_output = vit.embeddings(pixel_values)

        # Expand prompt tokens to batch dimension and apply dropout
        prompts = self.prompt_dropout(
            self.prompt_embeddings.expand(batch_size, -1, -1)
        )

        # Insert prompts between CLS and patch tokens.
        # Final layout: [CLS | p1...pN | patch1...patch196]
        # CLS stays at index 0 so the classifier head needs no changes.
        embedding_output = torch.cat([
            embedding_output[:, :1, :],   # CLS token  (B, 1, 768)
            prompts,                       # prompt tokens (B, N, 768)
            embedding_output[:, 1:, :],   # 196 patch tokens (B, 196, 768)
        ], dim=1)
        # Shape: (B, 1 + num_prompts + 196, 768)

        # Full transformer encoder — handles any sequence length
        encoder_outputs = vit.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

        # Layer norm then classify from CLS token at index 0
        sequence_output = vit.layernorm(encoder_outputs.last_hidden_state)
        cls_output = sequence_output[:, 0, :]
        logits = self.vit_model.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )

    def count_prompt_parameters(self) -> int:
        """I return the number of trainable prompt parameters."""
        return self.prompt_embeddings.numel()
