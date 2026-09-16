"""Audited inference-only prefix helpers, copied without training worker initialization.
Source: DualISL_Train_RewardV4/dual_isl_train/workers/qwen_voice_design.py
Source SHA256: a8b21667b63589d9d88226b3e5f7e3da343a6982c78a6634e2958c8c992e4c26
These helpers construct non-streaming VoiceDesign conditions, no reference audio.
"""
from __future__ import annotations

class VoiceDesignConditioning:
    def _tokenize_request(self, request: dict[str, str]):
            input_id = self.wrapper._tokenize_texts([self.wrapper._build_assistant_text(request["text"])])[0]
            instruct = request.get("instruct", "")
            instruct_id = self.wrapper._tokenize_texts([self.wrapper._build_instruct_text(instruct)])[0] if instruct else None
            return input_id, instruct_id

    def _conditioning(self, request: dict[str, str], *, talker=None):
            """Reproduce the non-streaming VoiceDesign prefix used by Qwen inference."""
            torch = self.torch
            core = self.core
            talker = talker or core.talker
            input_id, instruct_id = self._tokenize_request(request)
            input_id = input_id.to(self.device)
            parts = []
            if instruct_id is not None:
                instruct_id = instruct_id.to(self.device)
                parts.append(talker.text_projection(talker.get_text_embeddings()(instruct_id)))
            language = request.get("language", "Auto")
            if language.lower() == "auto":
                language_id = None
            else:
                language_id = core.config.talker_config.codec_language_id[language.lower()]
            special = torch.tensor(
                [[core.config.tts_bos_token_id, core.config.tts_eos_token_id, core.config.tts_pad_token_id]],
                device=self.device, dtype=input_id.dtype,
            )
            tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(special)).chunk(3, dim=1)
            if language_id is None:
                codec_prefill = [[
                    core.config.talker_config.codec_nothink_id,
                    core.config.talker_config.codec_think_bos_id,
                    core.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill = [[
                    core.config.talker_config.codec_think_id,
                    core.config.talker_config.codec_think_bos_id,
                    language_id,
                    core.config.talker_config.codec_think_eos_id,
                ]]
            codec_ids = torch.tensor(codec_prefill, device=self.device, dtype=input_id.dtype)
            codec_embed_0 = talker.get_input_embeddings()(codec_ids)
            codec_embed_1 = talker.get_input_embeddings()(torch.tensor(
                [[core.config.talker_config.codec_pad_id, core.config.talker_config.codec_bos_id]],
                device=self.device, dtype=input_id.dtype,
            ))
            codec_embed = torch.cat([codec_embed_0, codec_embed_1], dim=1)
            role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
            prefix_codec = torch.cat([
                tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed,
            ], dim=1) + codec_embed[:, :-1]
            prefix = torch.cat([role, prefix_codec], dim=1)
            text_part = torch.cat([
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed,
            ], dim=1)
            text_part = text_part + talker.get_input_embeddings()(torch.full(
                (1, text_part.shape[1]), core.config.talker_config.codec_pad_id,
                device=self.device, dtype=input_id.dtype,
            ))
            codec_bos = tts_pad_embed + talker.get_input_embeddings()(torch.tensor(
                [[core.config.talker_config.codec_bos_id]], device=self.device, dtype=input_id.dtype,
            ))
            prefix = torch.cat([prefix, text_part, codec_bos], dim=1)
            parts.append(prefix)
            return torch.cat(parts, dim=1), tts_pad_embed

    def _frame_embeddings(self, codes, tts_pad_embed, *, talker=None):
            talker = talker or self.core.talker
            # Match Qwen generation's cat(...).sum(1) reduction order exactly.  A
            # left-associated BF16 sum is measurably different and breaks replay.
            pieces = [talker.get_input_embeddings()(codes[:, :, 0])]
            pieces.extend(
                talker.code_predictor.get_input_embeddings()[codebook - 1](codes[:, :, codebook])
                for codebook in range(1, 16)
            )
            return self.torch.stack(pieces, dim=2).sum(2) + tts_pad_embed
