import json
import time
import uuid
import torch
import uvicorn
import os
from uvicorn.config import Config
from uvicorn import Server
from pydantic import BaseModel, Field
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
from typing import Any, Dict
from coreai.serve.api_schema import (
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    ModelPermission,
    CompletionUsage,
)
import requests
from PIL import Image
import base64
from io import BytesIO
from loguru import logger
import uuid

_TEXT_COMPLETION_CMD = object()

global_model = None
source_prefix = "You are a helpful assistant."
local_doc_qa = None
conv = None
debug_mode = False


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image_data = base64.b64decode(image_file)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


async def text_complete_last_message_vllm(
    history, stop_words_ids, gen_kwargs, tokenizer, model, request_id
):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[: -len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    results_generator = model.generate(prompt, gen_kwargs, request_id)
    output = ""
    async for request_output in results_generator:
        p = request_output.prompt
        output = request_output.outputs[-1].text
    # assert output.startswith(prompt)
    # output = output[len(prompt) :]
    output = trim_stop_words(output, ["<|endoftext|>", im_end])
    # print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output


@app.get("/")
async def root():
    return {"message": "Hello World, did you using frp get your local service out?"}


@app.get("/v1/models")
async def show_available_models():
    # global model_name_runing
    models = ["namo-500m", "namo-700m", "gpt4o", "o1", "internvl2-8b"]
    models.sort()
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


def get_response_msg_auto_stream(msg, model_id, stream=False):
    if stream:
        gen = get_info_msg_stream(msg, model_id=model_id)
        return StreamingResponse(gen, media_type="text/event-stream")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=msg),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=time.time_ns() // 1_000_000,
        model=model_id,
        choices=[choice_data],
        object="chat.completion",
    )


async def get_info_msg_stream(content: str, model_id: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(content=content), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=time.time_ns() // 1_000_000,
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "data: {}\n\n".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=time.time_ns() // 1_000_000,
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "data: {}\n\n".format(chunk.model_dump_json(exclude_unset=True))


def _map_content(content):
    if isinstance(content, str):
        return content, None
    else:
        if len(content) > 1 and any(item.type == "image_url" for item in content):
            # might contains image
            # print(content)
            text = next(itm for itm in content if itm.type == "text").text
            img = next(itm.image_url for itm in content if itm.type == "image_url").url
            return "<image> " + text, load_image(img)
        else:
            return content[0], None


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global global_model, source_prefix, local_doc_qa, conv, debug_mode

    t_id = int(time.time())
    r_id = f"chatcmpl-{t_id}"

    if request.stream:
        response = stream_response(request, gen_kwargs=None)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )
    # else:
    text = global_model.chat_with_request(
        request.model_dump()["messages"], stream=False
    )
    vis_chat_resp = {
        "id": r_id,
        "object": "chat.completion",  # chat.completions.chunk for stream
        "created": t_id,
        # "model": global_model.model_name,
        "model": "namo",
        "system_fingerprint": "fp_111111111",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    logger.debug(f"Response: {vis_chat_resp}")
    return vis_chat_resp


def stream_response(
    request,
    gen_kwargs: Dict[str, Any],
):

    # prompt_txt_num = len(gen_kwargs["inputs"])
    prompt_txt_num = 10
    all_output = ""
    response_generator = global_model.stream_chat_with_request(
        request.model_dump()["messages"]
    )

    for new_text in response_generator:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=new_text),
            finish_reason=None,
        )
        if new_text is not None:
            chunk = ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=time.time_ns() // 1_000_000,
                model="namo",
                choices=[choice_data],
                object="chat.completion.chunk",
            )
            # print(chunk.model_dump_json())
            # print(new_text)
            all_output += new_text
            yield "data: {}\n\n".format(chunk.model_dump_json())

    completion_txt_num = len(all_output)
    # recalculate token
    prompt_txt_num *= 1.33
    completion_txt_num *= 1.33

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant", content=""), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=time.time_ns() // 1_000_000,
        model="namo",
        choices=[choice_data],
        object="chat.completion.chunk",
        usage=CompletionUsage(
            prompt_tokens=int(prompt_txt_num),
            completion_tokens=int(completion_txt_num),
            total_tokens=int(completion_txt_num + prompt_txt_num),
        ),
    )
    yield "data: {}\n\n".format(chunk.model_dump_json())


def start_server(model="namo", ip="127.0.0.1", port=8080):
    global global_model

    if not os.path.exists(model):
        if model == "minicpm":
            model_path = "checkpoints/minicpm_v2_6"
        elif model == "internvl2":
            model_path = "checkpoints/internvl2-8b/"
        elif model == "qwen2vl":
            model_path = "checkpoints/qwen2-vl-7b/"
        else:
            model_path = "checkpoints/Namo-500M-V1/"
    else:
        model_path = model

    if "internvl" in model_path:
        logger.warning("not supported for now")
    elif "minicpm" in model_path:
        logger.warning("not supported for now")
    elif "qwen2-vl" in model_path:
        logger.warning("not supported for now")
    elif "namo" in model_path.lower():
        from namo.api.namo import NamoVL

        global_model = NamoVL(model_path=model_path, device="auto")
        logger.success("namo model initiated!")
    else:
        ValueError(f"unsupported model: {model_path}")

    http_config = Config(app=app, host=ip, port=port, log_level="info")
    http_server = Server(config=http_config)

    import asyncio

    uvicorn.run(app, host=ip, port=port, workers=1)
