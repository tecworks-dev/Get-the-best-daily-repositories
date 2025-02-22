import torch
from namo.models.symbols import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    AUDIO_TOKEN_INDEX,
)
from namo.utils.process_utils import unpad_image, get_anyres_image_grid_shape
from .meta_vision import NamoMetaVisionForCausalLM


class NamoMetaOmniForCausalLM(NamoMetaVisionForCausalLM):
    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def get_audio_projector(self):
        return self.get_model().audio_projector

    def encode_audio(self, audio, audio_lengths):
        audio_encoder_type = self.config.audio_encoder_type
        audio_encoder = self.get_audio_encoder()
        if "whisper" in audio_encoder_type.lower():
            encoder_outs = audio_encoder(audio.permute(0, 2, 1))
            audio_lengths = (audio_lengths + 1) // 2
        else:
            raise ValueError(f"Unknown audio encoder: {audio_encoder}")
        audio_projector_type = self.config.audio_projector_type
        audio_projector = self.get_audio_projector()
        if audio_projector_type == "linear":
            encoder_outs = audio_projector(encoder_outs)
            audio_lengths = audio_lengths // audio_projector.k
        else:
            raise ValueError(f"Unknown audio projector: {audio_projector_type}")
        audio_features = [
            encoder_outs[i, : audio_lengths[i]] for i in range(len(encoder_outs))
        ]
        return audio_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
        audio=None,
        audio_lengths=None,
        pixel_attention_mask=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        audio_encoder = self.get_audio_encoder()
        if audio_encoder is None or audio_encoder is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            (
                                num_patch_width,
                                num_patch_height,
                            ) = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size,
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(
                images, pixel_attention_mask=pixel_attention_mask
            )

        # preparing audio features
        audio_features = self.encode_audio(audio, audio_lengths)

        # for video test
        # image_features = self.temporal_aggregation(image_features)
        # print(f'imgfahture: {image_features.shape}')

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_conn_ve_llm", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # I->T, A->T, T->T, AI->T, TI->T, AT->T, AIT->T
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_audio_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # no image or audio, pure text.
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audio_frames = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            if num_images == 0 and num_audio_frames == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [
                        cur_input_embeds_1,
                        cur_image_features[0:0],
                        cur_audio_features[0:0],
                    ],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_audio_idx += 1
                continue

            image_audio_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == IMAGE_TOKEN_INDEX)
                    | (cur_input_ids == AUDIO_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim_noau = []
            cur_labels = labels[batch_idx]
            cur_labels_noim_noau = []
            for i in range(len(image_audio_token_indices) - 1):
                cur_input_ids_noim_noau.append(
                    cur_input_ids[
                        image_audio_token_indices[i]
                        + 1 : image_audio_token_indices[i + 1]
                    ]
                )
                cur_labels_noim_noau.append(
                    cur_labels[
                        image_audio_token_indices[i]
                        + 1 : image_audio_token_indices[i + 1]
                    ]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim_noau]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim_noau)
            )
            cur_input_embeds_no_im_no_au = torch.split(
                cur_input_embeds, split_sizes, dim=0
            )
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + num_audio_frames + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im_no_au[i])
                cur_new_labels.append(cur_labels_noim_noau[i])
                if i < num_images + num_audio_frames:
                    if (
                        cur_input_ids[image_audio_token_indices[i + 1]]
                        == IMAGE_TOKEN_INDEX
                    ):
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    elif (
                        cur_input_ids[image_audio_token_indices[i + 1]]
                        == AUDIO_TOKEN_INDEX
                    ):
                        cur_audio_features = audio_features["inputs_embeds"][
                            cur_audio_idx
                        ]
                        cur_audio_idx += 1
                        cur_new_input_embeds.append(cur_audio_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_audio_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    else:
                        raise ValueError

            if num_images != 0 and num_audio_frames == 0:
                cur_audio_features = audio_features["inputs_embeds"][cur_audio_idx]
                cur_audio_idx += 1
                cur_new_input_embeds.append(cur_audio_features[0:0])
            elif num_images == 0 and num_audio_frames != 0:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features[0:0])
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        assert cur_image_idx == image_features.shape[0]
        assert cur_audio_idx == audio_features["inputs_embeds"].shape[0]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens(
                [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_AUDIOTOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_conn_ve_llm:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_conn_ve_llm:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
