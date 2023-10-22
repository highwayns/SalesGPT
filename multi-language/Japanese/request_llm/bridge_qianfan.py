
import time, requests, json
from multiprocessing import Process, Pipe
from functools import wraps
from datetime import datetime, timedelta
from toolbox import get_conf, update_ui, is_any_api_key, select_api_key, what_keys, clip_history, trimmed_format_exc, get_conf

model_name = '千帆大模型プラットフォーム'
timeout_bot_msg = '[Local Message] Request timeout. Network error.'

def cache_decorator(timeout):
    cache = {}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, frozenset(kwargs.items()))
            # Check if result is already cached and not expired
            if key in cache:
                result, timestamp = cache[key]
                if datetime.now() - timestamp < timedelta(seconds=timeout):
                    return result

            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[key] = (result, datetime.now())
            return result
        return wrapper
    return decorator

@cache_decorator(timeout=3600)
def get_access_token():
    """
    AKを使用するします，SKで認証署名を生成します（Access Token）
    :return: access_token，またはNone(エラーが発生した場合)
    """
    # if (access_token_cache is None) or (time.time() - last_access_token_obtain_time > 3600):
    BAIDU_CLOUD_API_KEY, BAIDU_CLOUD_SECRET_KEY = get_conf('BAIDU_CLOUD_API_KEY', 'BAIDU_CLOUD_SECRET_KEY')

    if len(BAIDU_CLOUD_SECRET_KEY) == 0: raise RuntimeError("テキストの翻訳")
    if len(BAIDU_CLOUD_API_KEY) == 0: raise RuntimeError("テキストの翻訳")

    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": BAIDU_CLOUD_API_KEY, "client_secret": BAIDU_CLOUD_SECRET_KEY}
    access_token_cache = str(requests.post(url, params=params).json().get("access_token"))
    return access_token_cache
    # else:
    #     return access_token_cache


def generate_message_payload(inputs, llm_kwargs, history, system_prompt):
    conversation_cnt = len(history) // 2
    if system_prompt == "": system_prompt = "Hello"
    messages = [{"role": "user", "content": system_prompt}]
    messages.append({"role": "assistant", "content": 'Certainly!'})
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index] if history[index]!="" else "Hello"
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1] if history[index]!="" else "Hello"
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "": continue
                if what_gpt_answer["content"] == timeout_bot_msg: continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']
    what_i_ask_now = {}
    what_i_ask_now["role"] = "user"
    what_i_ask_now["content"] = inputs
    messages.append(what_i_ask_now)
    return messages


def generate_from_baidu_qianfan(inputs, llm_kwargs, history, system_prompt):
    BAIDU_CLOUD_QIANFAN_MODEL,  = get_conf('BAIDU_CLOUD_QIANFAN_MODEL')

    url_lib = {
        "ERNIE-Bot":            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions" ,
        "ERNIE-Bot-turbo":      "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"  ,
        "BLOOMZ-7B":            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1",

        "Llama-2-70B-Chat":     "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_70b",
        "Llama-2-13B-Chat":     "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_13b",
        "Llama-2-7B-Chat":      "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_7b",
    }

    url = url_lib[BAIDU_CLOUD_QIANFAN_MODEL]

    url += "?access_token=" + get_access_token()


    payload = json.dumps({
        "messages": generate_message_payload(inputs, llm_kwargs, history, system_prompt),
        "stream": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload, stream=True)
    buffer = ""
    for line in response.iter_lines():
        if len(line) == 0: continue
        try:
            dec = line.decode().lstrip('data:')
            dec = json.loads(dec)
            incoming = dec['result']
            buffer += incoming
            yield buffer
        except:
            if ('error_code' in dec) and ("max length" in dec['error_msg']):
                raise ConnectionAbortedError(dec['error_msg'])  # コンテキストが長すぎてトークンがオーバーフローする
            elif ('error_code' in dec):
                raise RuntimeError(dec['error_msg'])


def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        ⭐マルチスレッドメソッド
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    watch_dog_patience = 5
    response = ""

    for response in generate_from_baidu_qianfan(inputs, llm_kwargs, history, sys_prompt):
        if len(observe_window) >= 1:
            observe_window[0] = response
        if len(observe_window) >= 2:
            if (time.time()-observe_window[1]) > watch_dog_patience: raise RuntimeError("原始文本。")
    return response

def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        ⭐シングルスレッドのテキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    chatbot.append((inputs, ""))

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    yield from update_ui(chatbot=chatbot, history=history)
    # テキストの翻訳
    try:
        for response in generate_from_baidu_qianfan(inputs, llm_kwargs, history, system_prompt):
            chatbot[-1] = (inputs, response)
            yield from update_ui(chatbot=chatbot, history=history)
    except ConnectionAbortedError as e:
        from .bridge_all import model_info
        if len(history) >= 2: history[-1] = ""; history[-2] = "" # 現在のオーバーフロー入力をクリアする：history[-2] 是本次输入, history[-1] 今回の出力です
        history = clip_history(inputs=inputs, history=history, tokenizer=model_info[llm_kwargs['llm_model']]['tokenizer'], 
                    max_token_limit=(model_info[llm_kwargs['llm_model']]['max_token'])) # 原始文本
        chatbot[-1] = (chatbot[-1][0], "[Local Message] 長さを短くしてください。入力が長すぎます, テキストの翻訳, もう一度お試しください。 (もう一度失敗した場合、入力が長すぎる可能性が高いです)")
        yield from update_ui(chatbot=chatbot, history=history, msg="例外") # 画面をリフレッシュする
        return
    
    # 要約出力
    response = f"[Local Message]: {model_name}応テキストの翻訳異常 ..."
    if response == f"[Local Message]: 待機中{model_name}原始文本 ...":
        response = f"[Local Message]: {model_name}応テキストの翻訳異常 ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)