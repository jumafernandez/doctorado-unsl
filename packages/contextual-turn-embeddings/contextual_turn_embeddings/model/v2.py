"""v2 — **espejo fiel de ``modeling_bert.py``** aplicado a turnos de diálogo.

Replica la **jerarquía OOP de BERT** (función ``eager_attention_forward``;
``BertTurn{SelfAttention,SelfOutput,Attention,Intermediate,Output,Layer,Encoder,
Embeddings}``; ``BertTurnPredictionHead*``; y la jerarquía
``BertTurnPreTrainedModel → BertTurnModel → ContextualTurnModelV2``, análoga a
``BertPreTrainedModel → BertModel → BertForMaskedLM``).

**Toda** diferencia respecto del BERT canónico está marcada inline con ``# DIFF`` y
catalogada en ``docs/model/v2.md``. Lo demás se mantiene *verbatim*.

``ContextualTurnModelV2`` conserva **la misma interfaz pública que el v1** (``forward``,
``encode``, ``save_pretrained``/``from_pretrained``, ``mask_embedding``,
``reconstruction_head``, ``next_turn_head``, ``speaker_embedding``) → es *drop-in* para
``train.compute_objectives`` y ``encode.encode_dialogues``.

Referencia: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..utils import load_safetensors, read_json, save_safetensors, write_json

__all__ = ["ContextualTurnModelV2", "BertTurnModel"]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TRAINING_ARGS_NAME = "training_args.json"

# FIEL: subconjunto de transformers.activations.ACT2FN. nn.GELU() (erf) == ACT2FN["gelu"].
ACT2FN = {"gelu": nn.GELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}


def _act(name: Any) -> nn.Module:
    return ACT2FN[name] if isinstance(name, str) else name


class BertTurnConfig:
    """Config estilo ``BertConfig`` derivada de ``ModelConfig``.

    Los módulos ``BertTurn*`` leen ``config.hidden_size``, ``config.num_attention_heads``,
    etc. *verbatim* como en BERT. Este objeto deriva esos nombres del ``ModelConfig``.

    # DIFF (plumbing): no es ``transformers.BertConfig``; mapeo 1:1 de los campos usados.
    # DIFF (turno≠token): ``max_position_embeddings``=``max_turns``, ``type_vocab_size``=``num_speakers``,
    #   y campos extra ``input_dim`` / ``use_speaker_embeddings`` / ``head_transform`` / ``is_causal``.
    """

    def __init__(self, m: ModelConfig):
        self.hidden_size = m.hidden_dim
        self.num_attention_heads = m.num_heads
        self.intermediate_size = m.ff_dim
        self.hidden_act = m.activation
        self.hidden_dropout_prob = m.dropout
        self.attention_probs_dropout_prob = m.dropout
        self.max_position_embeddings = m.max_turns          # DIFF: turnos (BERT: 512)
        self.type_vocab_size = m.num_speakers               # DIFF: speakers (BERT: segmentos)
        self.num_hidden_layers = m.num_layers
        self.layer_norm_eps = 1e-12                         # FIEL a BERT (v1 usaba 1e-5)
        self.initializer_range = 0.02                       # FIEL a BERT
        # campos turno-específicos (no existen en BertConfig):
        self.input_dim = m.input_dim
        self.use_speaker_embeddings = m.use_speaker_embeddings
        self.head_transform = m.head_transform
        self.is_causal = m.attention_mode == "autoregressive"  # DIFF #7


# --------------------------------------------------------------------------- #
# Atención
# --------------------------------------------------------------------------- #
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FIEL a ``eager_attention_forward`` de ``modeling_bert.py``."""
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class BertTurnSelfAttention(nn.Module):
    """FIEL a ``BertSelfAttention`` (camino *eager*).

    # DIFF (plumbing): sin KV-cache ni despacho ``ALL_ATTENTION_FUNCTIONS`` — solo eager.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self, query_layer, key_layer, value_layer, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout.p,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class BertTurnSelfOutput(nn.Module):
    """FIEL a ``BertSelfOutput``: dense -> dropout -> LayerNorm(h + input) [post-LN]."""

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertTurnAttention(nn.Module):
    """FIEL a ``BertAttention``.

    # DIFF (plumbing): sin cross-attention (``is_cross_attention``) ni ``prune_heads``.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.self = BertTurnSelfAttention(config)
        self.output = BertTurnSelfOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.self(hidden_states, attention_mask)
        attention_output = self.output(attn_output, hidden_states)
        return attention_output, attn_weights


class BertTurnIntermediate(nn.Module):
    """FIEL a ``BertIntermediate``: dense(hidden->ff) -> activación."""

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = _act(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertTurnOutput(nn.Module):
    """FIEL a ``BertOutput``: dense(ff->hidden) -> dropout -> LayerNorm(h + input)."""

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertTurnLayer(nn.Module):
    """FIEL a ``BertLayer`` (incluye ``feed_forward_chunk``).

    # DIFF (plumbing): sin cross-attention, sin ``apply_chunking_to_forward``
    #   (``chunk_size_feed_forward=0`` ⇒ no-op) ni ``GradientCheckpointingLayer``.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.attention = BertTurnAttention(config)
        self.intermediate = BertTurnIntermediate(config)
        self.output = BertTurnOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output, _ = self.attention(hidden_states, attention_mask)
        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertTurnEncoder(nn.Module):
    """FIEL a ``BertEncoder``: pila secuencial de ``BertTurnLayer``."""

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertTurnLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertTurnEmbeddings(nn.Module):
    """ADAPTADO de ``BertEmbeddings``.

    # DIFF #1: entrada **continua** ``inputs_embeds`` (``e_t``) + ``input_proj`` (Linear si
    #   ``input_dim != hidden_size``), en vez de ``word_embeddings(input_ids)``.
    # DIFF #2: ``token_type_embeddings`` -> ``speaker_embeddings``.
    # DIFF (plumbing): sin buffers ``position_ids`` / ``token_type_ids`` registrados.
    Orden FIEL a BERT: (inputs_embeds + speaker) + position -> LayerNorm -> dropout.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.input_proj: nn.Module = (
            nn.Linear(config.input_dim, config.hidden_size)
            if config.input_dim != config.hidden_size
            else nn.Identity()
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.speaker_embeddings: Optional[nn.Embedding] = (
            nn.Embedding(config.type_vocab_size, config.hidden_size)
            if config.use_speaker_embeddings
            else None
        )
        self.num_speakers = config.type_vocab_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, inputs_embeds: torch.Tensor, speaker_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_length = inputs_embeds.shape[1]
        embeddings = self.input_proj(inputs_embeds)
        if self.speaker_embeddings is not None and speaker_ids is not None:
            ids = speaker_ids.clamp(min=0, max=self.num_speakers - 1)
            embeddings = embeddings + self.speaker_embeddings(ids)
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0)
        embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertTurnPredictionHeadTransform(nn.Module):
    """FIEL a ``BertPredictionHeadTransform``: dense -> activación -> LayerNorm."""

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = _act(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertTurnLMPredictionHead(nn.Module):
    """ADAPTADO de ``BertLMPredictionHead`` (transform + decoder + bias).

    # DIFF #4/#6: el ``decoder`` produce un **embedding continuo** (``out_dim`` = ``input_dim``)
    #   en vez de logits sobre vocab, y **no** se ata a ``word_embeddings`` (no hay vocab de
    #   turnos en Fase 1). El ``transform`` (FIEL) es opcional vía ``head_transform``.
    """

    def __init__(self, config: BertTurnConfig, out_dim: int):
        super().__init__()
        self.transform: nn.Module = (
            BertTurnPredictionHeadTransform(config) if config.head_transform else nn.Identity()
        )
        self.decoder = nn.Linear(config.hidden_size, out_dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(out_dim))  # FIEL: bias propio (como BertLMPredictionHead)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# Jerarquía de modelos (espeja PreTrainedModel -> BertModel -> BertForMaskedLM)
# --------------------------------------------------------------------------- #
class BertTurnPreTrainedModel(nn.Module):
    """Base con ``config`` + ``_init_weights`` (FIEL a ``BertPreTrainedModel._init_weights``).

    # DIFF (plumbing): no es un ``transformers.PreTrainedModel`` (sin from_pretrained del Hub,
    #   sin device_map, etc.); solo sostiene la config y la inicialización de pesos de BERT.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, BertTurnLMPredictionHead):
            module.bias.data.zero_()  # FIEL: BertPreTrainedModel hace init.zeros_(head.bias)

    def init_weights(self) -> None:
        self.apply(self._init_weights)


class BertTurnModel(BertTurnPreTrainedModel):
    """FIEL a ``BertModel`` (embeddings + encoder).

    # DIFF #5: sin ``pooler`` (no se usa).
    # DIFF (plumbing): la máscara se arma con la semántica de ``get_extended_attention_mask``
    #   (HF reciente la delega a ``masking_utils``); ``forward`` devuelve el tensor
    #   ``sequence_output`` en vez de un ``BaseModelOutputWithPooling``.
    """

    def __init__(self, config: BertTurnConfig):
        super().__init__(config)
        self.embeddings = BertTurnEmbeddings(config)
        self.encoder = BertTurnEncoder(config)
        self.init_weights()

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, seq_len: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Máscara aditiva: 0 = permitido, ``finfo.min`` = prohibido (padding y, si AR, causal)."""
        device = attention_mask.device
        allowed = attention_mask[:, None, None, :].bool()  # [B,1,1,S] keys no-padding
        if self.config.is_causal:  # DIFF #7
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            allowed = allowed & causal[None, None]  # [B,1,S,S]
        ext = torch.zeros(allowed.shape, dtype=dtype, device=device)
        return ext.masked_fill(~allowed, torch.finfo(dtype).min)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding_output = self.embeddings(inputs_embeds, speaker_ids)
        ext_mask = self.get_extended_attention_mask(
            attention_mask, inputs_embeds.shape[1], embedding_output.dtype
        )
        sequence_output = self.encoder(embedding_output, ext_mask)
        return sequence_output


class ContextualTurnModelV2(BertTurnPreTrainedModel):
    """Encoder contextual de turnos (``f2``), análogo a ``BertForMaskedLM`` envolviendo ``BertModel``.

    Conserva la interfaz pública del v1 (drop-in para ``train``/``encode``).
    # DIFF: en vez de la MLM head con softmax sobre vocab, expone heads continuos
    #   (``reconstruction_head`` / ``next_turn_head``) que consume ``train.compute_objectives``.
    """

    def __init__(self, config: ModelConfig):
        bert_config = BertTurnConfig(config)
        super().__init__(bert_config)
        # ``self.config`` debe ser el ModelConfig (lo usan compute_objectives/encode/save):
        self.config = config
        self.bert_config = bert_config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim or config.input_dim

        self.bert = BertTurnModel(bert_config)  # inicializa sus propios pesos
        # h_t vive en hidden_size; proyectar solo si difiere (FIEL: BERT usa el hidden directo).
        self.output_proj: nn.Module = (
            nn.Linear(bert_config.hidden_size, self.output_dim)
            if bert_config.hidden_size != self.output_dim
            else nn.Identity()
        )
        # Heads que consume train.compute_objectives (análogos a la MLM head, salida continua):
        self.reconstruction_head = BertTurnLMPredictionHead(bert_config, self.input_dim)
        self.next_turn_head = BertTurnLMPredictionHead(bert_config, self.input_dim)
        # DIFF: vector [MASK] aprendido en espacio de entrada (BERT enmascara ids de token).
        self.mask_embedding = nn.Parameter(torch.empty(self.input_dim))
        nn.init.normal_(self.mask_embedding, std=0.02)

        # init de los módulos propios (el bert ya se inicializó solo):
        self.output_proj.apply(self.bert._init_weights)
        self.reconstruction_head.apply(self.bert._init_weights)
        self.next_turn_head.apply(self.bert._init_weights)

    # Expuesto para encode.encode_dialogues (chequea ``model.speaker_embedding is not None``).
    @property
    def speaker_embedding(self) -> Optional[nn.Embedding]:
        return self.bert.embeddings.speaker_embeddings

    def forward(
        self,
        batch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = batch_embeddings.shape[1]
        if seq_len > self.config.max_turns:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_turns ({self.config.max_turns})"
            )
        sequence_output = self.bert(batch_embeddings, attention_mask, speaker_ids)
        # FIEL a BERT: NO aplicamos residual externo de salida (output_residual es solo de v1).
        # h_t sale libre del encoder, sin anclarse al e_t de entrada.
        return self.output_proj(sequence_output)

    @torch.no_grad()
    def encode(
        self,
        batch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward en modo eval (dropout off) devolviendo embeddings contextuales."""
        was_training = self.training
        self.eval()
        out = self.forward(batch_embeddings, attention_mask, speaker_ids)
        if was_training:
            self.train()
        return out

    # ------------------------------------------------------------------ #
    # Persistencia HF-style (mismo formato que v1: config.json + model.safetensors).
    # ------------------------------------------------------------------ #
    def save_pretrained(
        self, output_dir: str, training_args: Optional[Dict[str, Any]] = None
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        write_json(self.config.to_dict(), os.path.join(output_dir, CONFIG_NAME))
        save_safetensors(self.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        if training_args is not None:
            write_json(training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = "cpu") -> "ContextualTurnModelV2":
        config = ModelConfig.from_dict(read_json(os.path.join(model_dir, CONFIG_NAME)))
        model = cls(config)
        state = load_safetensors(os.path.join(model_dir, WEIGHTS_NAME), device="cpu")
        model.load_state_dict(state)
        model.to(device)
        return model
