"""
火山方舟 Doubao-Seed-2.0-lite 统一调用封装
官方SDK: volcengine-python-sdk[ark]
文档: https://www.volcengine.com/docs/82379/1263279
"""
import os
import re
import json
from volcenginesdkarkruntime import Ark

DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-2-lite-250520")
ARK_API_KEY  = os.getenv("ARK_API_KEY", "YOUR_ARK_API_KEY_HERE")


def get_ark_client() -> Ark:
    return Ark(api_key=ARK_API_KEY)


def chat_completion(messages, model=None, temperature=0.7, max_tokens=1024, stream=False):
    if model is None:
        model = DOUBAO_MODEL
    client = get_ark_client()
    if stream:
        response_text = ""
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text
    completion = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return completion.choices[0].message.content


def chat_completion_json(messages, model=None, temperature=0.3):
    if model is None:
        model = DOUBAO_MODEL
    msgs = [m.copy() for m in messages]
    injected = False
    for msg in msgs:
        if msg["role"] == "system":
            msg["content"] += "\n\n【重要】只输出纯JSON，不要有任何额外文字、代码块标记或解释。"
            injected = True
            break
    if not injected:
        msgs = [{"role": "system", "content": "只输出纯JSON，不要有任何额外文字或代码块标记。"}] + msgs
    client = get_ark_client()
    completion = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, max_tokens=4096)
    return completion.choices[0].message.content
