
import time
import threading
import importlib
from toolbox import update_ui, get_conf, update_ui_lastest_msg
from multiprocessing import Process, Pipe

model_name = '原始文本'

def validate_key():
    XFYUN_APPID,  = get_conf('XFYUN_APPID', )
    if XFYUN_APPID == '00000000' or XFYUN_APPID == '': 
        return False
    return True

def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        ⭐マルチスレッドメソッド
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    watch_dog_patience = 5
    response = ""

    if validate_key() is False:
        raise RuntimeError('请配置讯飞星火大模型的XFYUN_APPID, XFYUN_API_KEY, XFYUN_API_SECRET')

    from .com_sparkapi import SparkRequestInstance
    sri = SparkRequestInstance()
    for response in sri.generate(inputs, llm_kwargs, history, sys_prompt):
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
    yield from update_ui(chatbot=chatbot, history=history)

    if validate_key() is False:
        yield from update_ui_lastest_msg(lastmsg="[Local Message]: 请配置讯飞星火大模型的XFYUN_APPID, XFYUN_API_KEY, XFYUN_API_SECRET", chatbot=chatbot, history=history, delay=0)
        return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    # テキストの翻訳    
    from .com_sparkapi import SparkRequestInstance
    sri = SparkRequestInstance()
    for response in sri.generate(inputs, llm_kwargs, history, system_prompt):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 要約出力
    if response == f"[Local Message]: 待機中{model_name}原始文本 ...":
        response = f"[Local Message]: {model_name}応テキストの翻訳異常 ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)