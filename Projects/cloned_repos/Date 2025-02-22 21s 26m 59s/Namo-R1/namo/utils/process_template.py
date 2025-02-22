import copy
import transformers
from namo.utils import convs as conversation_lib
import torch
from typing import Dict, Sequence
from namo.utils.process_utils import tokenizer_image_token
from namo.models.symbols import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from transformers import AutoTokenizer


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_qwen(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(f'conversations: {conversations}')
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    sep2 = conv.sep + conv.roles[0]  # <|im_end|><|im_start|>user\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        # due to spe2 will involve system, merge it to first round
        if len(rounds) > 1:
            rounds[0:2] = [sep2.join(rounds[0:2])]
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                elif len(rounds) > 1 and i == 0:
                    round_len = (
                        len(tokenizer_image_token(rou, tokenizer)) + 1
                    )  # for <|im_end|>
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    round_len = (
                        len(tokenizer_image_token(rou, tokenizer)) + 3
                    )  # for <|im_start|>user\n last round
                    instruction_len = (
                        len(tokenizer_image_token(parts[0], tokenizer)) + 3
                    )  # for <|im_start|>user\n
                else:
                    round_len = (
                        len(tokenizer_image_token(rou, tokenizer)) + 4
                    )  # for <|im_start|>user\n .. <|im_end|>
                    instruction_len = (
                        len(tokenizer_image_token(parts[0], tokenizer)) + 3
                    )  # for <|im_start|>user\n
            else:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                elif len(rounds) > 1 and i == 0:
                    round_len = len(tokenizer(rou).input_ids) + 1  # for <|im_end|>
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    round_len = (
                        len(tokenizer(rou).input_ids) + 3
                    )  # for <|im_start|>user\n last round
                    instruction_len = (
                        len(tokenizer(parts[0]).input_ids) + 3
                    )  # for <|im_start|>user\n
                else:
                    round_len = (
                        len(tokenizer(rou).input_ids) + 4
                    )  # for <|im_start|>user\n .. <|im_end|>
                    instruction_len = (
                        len(tokenizer(parts[0]).input_ids) + 3
                    )  # for <|im_start|>user\n

            # if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
            #     round_len -= 1
            #     instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                    f"{conversations}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # Note: LLama3 has bos while Qwen bos is null, we don't need add_special_token here. (Already added in template)
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(
                    prompt, tokenizer, return_tensors="pt", add_special_tokens=False
                )
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA3

    # Mask targets
    # <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    sep = f"<|eot_id|><|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n"
    # <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
    sep2 = f"<|eot_id|><|start_header_id|>{conv.roles[0]}<|end_header_id|>\n\n"
    # <|start_header_id|>assistant<|end_header_id|>\n\n [128006, 882, 128007, 271]
    # <|eot_id|> [128009]

    # print(targets)

    # [128000, 128009, 128006, 882, 128007, 271] <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
    # print(f'{tokenizer.encode(sep, add_special_tokens=False)} {sep}')
    # [128000, 128009, 128006, 78191, 128007, 271] <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    # print(f'{tokenizer.encode(sep2, add_special_tokens=False)} {sep2}')
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        # due to spe2 will involve system, merge it to first round
        if len(rounds) > 1:
            rounds[0:2] = [sep2.join(rounds[0:2])]
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX

        # print(f'rounds: ----> {rounds}')
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(
                        tokenizer_image_token(rou, tokenizer, add_special_tokens=False)
                    )
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=False
                        )
                    )
                elif len(rounds) > 1 and i == 0:
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=False
                            )
                        )
                        + 1
                    )  # for <|eot_id|>
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=False
                        )
                    )
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    # for <|start_header_id|>user<|end_header_id|>\n\n last round <|eot_id|> already have
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=False
                            )
                        )
                        + 4
                    )
                    instruction_len = (
                        len(
                            tokenizer_image_token(
                                parts[0], tokenizer, add_special_tokens=False
                            )
                        )
                        + 4
                    )  # for <|start_header_id|>user<|end_header_id|>\n\n
                else:
                    # for <|start_header_id|>user<|end_header_id|>\n\n .. <|eot_id|>
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=False
                            )
                        )
                        + 5
                    )
                    instruction_len = (
                        len(
                            tokenizer_image_token(
                                parts[0], tokenizer, add_special_tokens=False
                            )
                        )
                        + 4
                    )  # for <|start_header_id|>user<|end_header_id|>\n\n
            else:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                    instruction_len = len(
                        tokenizer(parts[0], add_special_tokens=False).input_ids
                    )
                elif len(rounds) > 1 and i == 0:
                    round_len = (
                        len(tokenizer(rou, add_special_tokens=False).input_ids) + 1
                    )
                    # for <|im_end|>
                    instruction_len = len(
                        tokenizer(parts[0], add_special_tokens=False).input_ids
                    )
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    # for <|im_start|>user\n last round
                    round_len = (
                        len(tokenizer(rou, add_special_tokens=False).input_ids) + 4
                    )
                    # for <|im_start|>user\n
                    instruction_len = (
                        len(tokenizer(parts[0], add_special_tokens=False).input_ids) + 4
                    )
                else:
                    # for <|im_start|>user\n .. <|im_end|>
                    round_len = (
                        len(tokenizer(rou, add_special_tokens=False).input_ids) + 5
                    )
                    # for <|im_start|>user\n
                    instruction_len = (
                        len(tokenizer(parts[0], add_special_tokens=False).input_ids) + 4
                    )

            # if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
            #     round_len -= 1
            #     instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mistral(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(
                    prompt, tokenizer, add_special_tokens=False, return_tensors="pt"
                )
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids

    targets = input_ids.clone()

    """
    <|im_start|>system
    You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
    hello<|im_end|><|im_start|>assistant
    am tallen! nice to meet u.<|im_end|><|im_start|>user
    Nice to see u, how dod u do?<|im_end|><|im_start|>assistant
    我很好，请问有什么可以帮你的吗？<|im_end|><|im_start|>user
    介绍一下你自己<|im_end|><|im_start|>assistant
    我是ChatGPT，一个人工智能助手<|im_end|><|im_start|>user
    你会python嘛？<|im_end|><|im_start|>assistant
    """

    """
    [INST] You should follow the instructions carefully and explain your answers in detail.
    hello [/INST] am tallen! nice to meet u.</s>[INST] Nice to see u, how dod u do? [/INST] 我很好，请问有什么可以帮你的吗？</s>[INST] 介绍一下你自己 [/INST] 我是ChatGPT，一个人工智能助手</s>
    
    [INST] You should follow the instructions carefully and explain your answers in detail.\n<image>\nPlease provide the bounding box coordinate of the region this sentence describes: dark suit near between tan and gray coat. [/INST] [0.28, 0.34, 0.4, 0.71]</s>[INST] Please provide a short description for this region: [0.56, 0.35, 0.67, 0.7]. [/INST] Third person from right.</s>[INST] Please provide a short description for this region: [0.66, 0.36, 0.79, 0.72]. [/INST] Red jacket.</s>[INST] Please provide a short description for this region: [0.66, 0.36, 0.79, 0.72]. [/INST] Red jacket.</s>[INST] Please provide a short description for this region: [0.56, 0.35, 0.67, 0.7]. [/INST] Third from right to left person.</s>[INST] Please provide a short description for this region: [0.36, 0.36, 0.5, 0.72]. [/INST] Tan jacket red scarf lady.</s>[INST] Please provide the bounding box coordinate of the region this sentence describes: guy on right. [/INST] [0.65, 0.31, 0.86, 0.74]</s>[INST] Please provide the bounding box coordinate of the region this sentence describes: third skier from rightbeing pointed at. [/INST] [0.56, 0.35, 0.67, 0.7]</s>
    """

    # print(f'conversations: {conversations}')
    # print(f'targets: {targets}')

    # Mask questions
    # sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    # sep2 = conv.sep + conv.roles[0]  # <|im_end|><|im_start|>user\n
    sep = conv.roles[1]  # '[/INST]'  [1032, 4, 1032] # Mistral空格会和后面的合并，坑！
    # sep2 = conv.sep2 # '</s>[INST]' [2, 3, 1032]
    sep2 = conv.sep2  # '</s>' [2, 3, 1032]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        rounds = [r for r in rounds if r != ""]
        # due to spe2 will involve system, merge it to first round
        # if len(rounds) > 1:
        #     rounds[0:2] = [sep2.join(rounds[0:2])]
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        # mask all answers
        for i, rou in enumerate(rounds):
            # rounds
            # [INST] You should follow the instructions carefully and explain your answers in detail.\nhello [/INST] am tallen! nice to meet u.
            # Nice to see u, how dod u do? [/INST] 我很好，请问有什么可以帮你的吗？
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # 补偿只有两种情况，最开头，和其他
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=False
                            )
                        )
                        + 1
                    )
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=False
                        )
                    )
                else:
                    # </s>分割，只需要补齐这个id即可，其他不影响
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=False
                            )
                        )
                        + 1
                    )
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=False
                        )
                    )
            else:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = (
                        len(tokenizer(rou, add_special_tokens=False).input_ids) + 1
                    )
                    instruction_len = len(
                        tokenizer(parts[0], add_special_tokens=False).input_ids
                    )
                else:
                    round_len = (
                        len(tokenizer(rou, add_special_tokens=False).input_ids) + 1
                    )
                    instruction_len = len(
                        tokenizer(parts[0], add_special_tokens=False).input_ids
                    )

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
            # print(target, cur_len)
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                    f"{conversations}"
                )

    # print(f'final target: {targets}')
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    add_special_tokens = False
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(
                    prompt,
                    tokenizer,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        ).input_ids

    targets = input_ids.clone()

    """
    You are a helpful assistant.<start_of_turn>user
    hello<end_of_turn>
    <start_of_turn>model
    am tallen! nice to meet u.<end_of_turn>
    <start_of_turn>user
    Nice to see u, how dod u do?<end_of_turn>
    <start_of_turn>model
    我很好，请问有什么可以帮你的吗？<end_of_turn>
    <start_of_turn>user
    介绍一下你自己<end_of_turn>
    <start_of_turn>model
    我是ChatGPT，一个人工智能助手<end_of_turn>
    <start_of_turn>user
    你会python嘛？<end_of_turn>
    <start_of_turn>model
    """

    # print(f'conversations: {conversations}')
    # print(f'targets: {targets}')

    # Mask questions
    sep = conv.roles[1]  # <start_of_turn>model\n 3
    sep2 = conv.sep + conv.roles[0]  # '<end_of_turn>\n<start_of_turn>user\n' 5
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        rounds = [r for r in rounds if r != ""]
        # due to spe2 will involve system, merge it to first round
        # if len(rounds) > 1:
        #     rounds[0:2] = [sep2.join(rounds[0:2])]
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        # mask all answers
        for i, rou in enumerate(rounds):
            # print(i, rou)
            # rounds
            # [INST] You should follow the instructions carefully and explain your answers in detail.\nhello [/INST] am tallen! nice to meet u.
            # Nice to see u, how dod u do? [/INST] 我很好，请问有什么可以帮你的吗？
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(
                        tokenizer_image_token(
                            rou, tokenizer, add_special_tokens=add_special_tokens
                        )
                    )
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=add_special_tokens
                        )
                    )
                elif len(rounds) > 1 and i == 0:
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=add_special_tokens
                            )
                        )
                        + 2
                    )  # for <end_of_turn>\n
                    instruction_len = len(
                        tokenizer_image_token(
                            parts[0], tokenizer, add_special_tokens=add_special_tokens
                        )
                    )
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    # for <|start_header_id|>user<|end_header_id|>\n\n last round <|eot_id|> already have
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=add_special_tokens
                            )
                        )
                        + 3
                    )
                    instruction_len = (
                        len(
                            tokenizer_image_token(
                                parts[0],
                                tokenizer,
                                add_special_tokens=add_special_tokens,
                            )
                        )
                        + 3
                    )  # for <|start_header_id|>user<|end_header_id|>\n\n
                else:
                    # for <|start_header_id|>user<|end_header_id|>\n\n .. <|eot_id|>
                    round_len = (
                        len(
                            tokenizer_image_token(
                                rou, tokenizer, add_special_tokens=add_special_tokens
                            )
                        )
                        + 5
                    )
                    instruction_len = (
                        len(
                            tokenizer_image_token(
                                parts[0],
                                tokenizer,
                                add_special_tokens=add_special_tokens,
                            )
                        )
                        + 4
                    )  # for <|start_header_id|>user<|end_header_id|>\n\n
            else:
                if len(rounds) == 1:
                    # no need to compensate
                    round_len = len(
                        tokenizer(rou, add_special_tokens=add_special_tokens).input_ids
                    )
                    instruction_len = len(
                        tokenizer(
                            parts[0], add_special_tokens=add_special_tokens
                        ).input_ids
                    )
                elif len(rounds) > 1 and i == 0:
                    round_len = (
                        len(
                            tokenizer(
                                rou, add_special_tokens=add_special_tokens
                            ).input_ids
                        )
                        + 2
                    )
                    # for <|im_end|>
                    instruction_len = len(
                        tokenizer(
                            parts[0], add_special_tokens=add_special_tokens
                        ).input_ids
                    )
                elif len(rounds) > 1 and i == len(rounds) - 1:
                    # for <|im_start|>user\n last round
                    round_len = (
                        len(
                            tokenizer(
                                rou, add_special_tokens=add_special_tokens
                            ).input_ids
                        )
                        + 3
                    )
                    # for <|im_start|>user\n
                    instruction_len = (
                        len(
                            tokenizer(
                                parts[0], add_special_tokens=add_special_tokens
                            ).input_ids
                        )
                        + 3
                    )
                else:
                    # for <|im_start|>user\n .. <|im_end|>
                    round_len = (
                        len(
                            tokenizer(
                                rou, add_special_tokens=add_special_tokens
                            ).input_ids
                        )
                        + 5
                    )
                    # for <|im_start|>user\n
                    instruction_len = (
                        len(
                            tokenizer(
                                parts[0], add_special_tokens=add_special_tokens
                            ).input_ids
                        )
                        + 4
                    )

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
            # print(target, cur_len)
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                    f"{conversations}"
                )

    # print(f'final target: {targets}')
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt().strip())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = conv.sep + conv.roles[1] + "\n"  # <start_of_turn>model\n
    round_sep = "\n" + conv.sep + conv.roles[0] + "\n"  # \n<start_of_turn>user\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(round_sep)
        cur_len = 1
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if i != 0:
                rou = round_sep + rou
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = (
                    len(tokenizer_image_token(rou, tokenizer)) - 1
                )  #  -1 for <bos>
                instruction_len = (
                    len(tokenizer_image_token(parts[0], tokenizer)) - 1
                )  # -1 for <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # -1 for <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # -1 for <bos>

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


if __name__ == "__main__":
    # conversation_lib.default_conversation = conversation_lib.conv_templates['mistral']
    # tokenizer_path = 'checkpoints/mistral-nemo-instruct/'
    conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    tokenizer_path = "checkpoints/gemma-2-9b-it/"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )
    print(tokenizer.pad_token_id)
    print(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.encode("<pad>")
    sources = [
        [
            {"from": "human", "value": "<video>\nWhat is happening in the video?"},
            {
                "from": "gpt",
                "value": "In the video, a man is sitting at a table with cups and glasses. He then proceeds to grab a cup of coffee and an apple while looking at his phone. Later on, he is seen looking at his phone again while sitting at the same table near a window. The man then has some cups in front of him and continues looking at his phone at the same table. Finally, he is seen looking at his phone again while sitting at the table.",
            },
            {
                "from": "human",
                "value": "Is the man interacting with any other objects or people?",
            },
            {
                "from": "gpt",
                "value": "The man is not seen interacting with any other objects or people in the video. He is just sitting at the table and using his phone.",
            },
            #   {
            #     "from": "human",
            #     "value": "Why do you think the man is looking at his phone so much?"
            #   },
            #   {
            #     "from": "gpt",
            #     "value": "It is not clear why the man is looking at his phone so often. We cannot infer his intentions as the video doesn't provide much context."
            #   }
        ]
    ]
    preprocess_mistral(sources, tokenizer, True)
