# httpsを参考にしました://github.com/GaiZhenbiao/ChuanhuChatGPT 项目

"""
    このファイルには主に2つの関数が含まれています

    マルチスレッドの機能を持たない関数：
    1. predict: 通常の会話時に使用するする，テキストの翻訳，マルチスレッドはできません

    原始文本
    2. predict_no_ui_long_connection：テキストの翻訳，テキストの翻訳，この関数はストリームのテキストの翻訳でこの問題を解決します，同様にマルチスレッドをサポートしています
"""

import os
import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib

# config_private.pyにはAPIやプロキシのURLなどの個人の秘密情報を入力してください
# 原始文本（gitの管理対象外），如果有，元のconfigファイルを上書きする
from toolbox import get_conf, update_ui, trimmed_format_exc, ProxyNetworkActivate
proxies, TIMEOUT_SECONDS, MAX_RETRY, ANTHROPIC_API_KEY = \
    get_conf('proxies', 'TIMEOUT_SECONDS', 'MAX_RETRY', 'ANTHROPIC_API_KEY')

timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
                  'ネットワークエラー，テキストの翻訳，プロキシの設定形式が正しいかどうか，原始文本[テキストの翻訳]://[アドレス]:[ポート]，欠かせない。'

def get_full_error(chunk, stream_response):
    """
        获取完整的从Openai返回的报错
    """
    while True:
        try:
            chunk += next(stream_response)
        except:
            break
    return chunk


def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=None, console_slience=False):
    """
    chatGPTに送信，返信待ち，一度に完了する，中間プロセスを表示しない。ただし、内部ではストリームを使用するして途中でネットワーク接続が切断されるのを防ぐ。
    inputs：
        今回の問い合わせの入力です
    sys_prompt:
        システムの静的プロンプト
    llm_kwargs：
        chatGPTの内部調整パラメータ
    history：
        以前の会話リストです
    observe_window = None：
        スレッドを越えて出力された部分を転送するために使用するされます，ほとんどの場合、単に見栄えの良い視覚効果のためだけです，空白のままにしておくことができます。observe_window[0]：観測ウィンドウ。observe_window[1]：ウォッチドッグ
    """
    from anthropic import Anthropic
    watch_dog_patience = 5 # テキストの翻訳, 5秒で設定できます
    prompt = generate_payload(inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    if len(ANTHROPIC_API_KEY) == 0:
        raise RuntimeError("ANTHROPIC_API_KEYオプションが設定されていません")

    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from .bridge_all import model_info
            anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
            # endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            # with ProxyNetworkActivate()
            stream = anthropic.completions.create(
                prompt=prompt,
                max_tokens_to_sample=4096,       # The maximum number of tokens to generate before stopping.
                model=llm_kwargs['llm_model'],
                stream=True,
                temperature = llm_kwargs['temperature']
            )
            break
        except Exception as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'タイムアウトしました，原始文本 ({retry}/{MAX_RETRY}) ……')
    result = ''
    try: 
        for completion in stream:
            result += completion.completion
            if not console_slience: print(completion.completion, end='')
            if observe_window is not None: 
                # 観測ウィンドウ，取得したデータを表示します
                if len(observe_window) >= 1: observe_window[0] += completion.completion
                # ウォッチドッグ，期限を超えて餌を与えない場合，それにより終了します
                if len(observe_window) >= 2:  
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("ユーザーがプログラムをキャンセルしました。")
    except Exception as e:
        traceback.print_exc()

    return result


def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
    chatGPTに送信，ストリーム形式で出力を取得する。
    テキストの翻訳。
    inputsは今回の問い合わせの入力です
    top_p, temperatureはchatGPTの内部チューニングパラメータです
    historyは以前の対話リストです（inputsまたはhistoryに関係なく注意してください，内容が長すぎると、トークンの数がオーバーフローするエラーが発生する）
    chatbotはWebUIで表示される対話リストです，それを変更する，そして出力してください，原始文本
    additional_fnはどのボタンがクリックされたかを表す，ボタンはfunctional.pyを参照してください
    """
    from anthropic import Anthropic
    if len(ANTHROPIC_API_KEY) == 0:
        chatbot.append((inputs, "ANTHROPIC_API_KEYが設定されていません"))
        yield from update_ui(chatbot=chatbot, history=history, msg="原始文本") # 画面をリフレッシュする
        return
    
    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    raw_input = inputs
    logging.info(f'[raw_input] {raw_input}')
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history, msg="原始文本") # 画面をリフレッシュする

    try:
        prompt = generate_payload(inputs, llm_kwargs, history, system_prompt, stream)
    except RuntimeError as e:
        chatbot[-1] = (inputs, f"提供されたAPIキーは要件を満たしていません，使用するできるものは含まれていません{llm_kwargs['llm_model']}のAPIキー。原始文本。")
        yield from update_ui(chatbot=chatbot, history=history, msg="APIキーが要件を満たしていない") # 画面をリフレッシュする
        return

    history.append(inputs); history.append("")

    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=True
            from .bridge_all import model_info
            anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
            # endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            # with ProxyNetworkActivate()
            stream = anthropic.completions.create(
                prompt=prompt,
                max_tokens_to_sample=4096,       # The maximum number of tokens to generate before stopping.
                model=llm_kwargs['llm_model'],
                stream=True,
                temperature = llm_kwargs['temperature']
            )
            
            break
        except:
            retry += 1
            chatbot[-1] = ((chatbot[-1][0], timeout_bot_msg))
            retry_msg = f"，原始文本 ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield from update_ui(chatbot=chatbot, history=history, msg="タイムアウトしました"+retry_msg) # 画面をリフレッシュする
            if retry > MAX_RETRY: raise TimeoutError

    gpt_replying_buffer = ""
    
    for completion in stream:
        try:
            gpt_replying_buffer = gpt_replying_buffer + completion.completion
            history[-1] = gpt_replying_buffer
            chatbot[-1] = (history[-2], history[-1])
            yield from update_ui(chatbot=chatbot, history=history, msg='正常') # 画面をリフレッシュする

        except Exception as e:
            from toolbox import regular_txt_to_markdown
            tb_str = '```\n' + trimmed_format_exc() + '```'
            chatbot[-1] = (chatbot[-1][0], f"[Local Message] 例外 \n\n{tb_str}")
            yield from update_ui(chatbot=chatbot, history=history, msg="Json例外" + tb_str) # 画面をリフレッシュする
            return
        



# https://github.com/jtsang4/claude-to-chatgpt/blob/main/claude_to_chatgpt/adapter.py
def convert_messages_to_prompt(messages):
    prompt = ""
    role_map = {
        "system": "Human",
        "user": "Human",
        "assistant": "Assistant",
    }
    for message in messages:
        role = message["role"]
        content = message["content"]
        transformed_role = role_map[role]
        prompt += f"\n\n{transformed_role.capitalize()}: {content}"
    prompt += "\n\nAssistant: "
    return prompt

def generate_payload(inputs, llm_kwargs, history, system_prompt, stream):
    """
    すべての情報を統合する，选择LLM模型，HTTPリクエストを生成する，リクエストの送信の準備をする
    """
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index]
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1]
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
    prompt = convert_messages_to_prompt(messages)

    return prompt


