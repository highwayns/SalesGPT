# httpsを参考にしました://github.com/GaiZhenbiao/ChuanhuChatGPT 项目

"""
    このファイルには主に3つの関数が含まれています

    マルチスレッドの機能を持たない関数：
    1. predict: 通常の会話時に使用するする，テキストの翻訳，マルチスレッドはできません

    原始文本
    2. predict_no_ui：高度な実験的な機能モジュールの呼び出し，画面にリアルタイムで表示されません，原始文本，マルチスレッドで並行処理できます，複雑な機能ロジックを簡単に実装する
    3. predict_no_ui_long_connection：テキストの翻訳，テキストの翻訳，この関数はストリームのテキストの翻訳でこの問題を解決します，同様にマルチスレッドをサポートしています
"""

import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib
import random

# config_private.pyにはAPIやプロキシのURLなどの個人の秘密情報を入力してください
# 原始文本（gitの管理対象外），如果有，元のconfigファイルを上書きする
from toolbox import get_conf, update_ui, is_any_api_key, select_api_key, what_keys, clip_history, trimmed_format_exc, is_the_upload_folder
proxies, TIMEOUT_SECONDS, MAX_RETRY, API_ORG = \
    get_conf('proxies', 'TIMEOUT_SECONDS', 'MAX_RETRY', 'API_ORG')

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

def decode_chunk(chunk):
    # 事前にいくつかの情報を読み取る （例外を判断するために使用するされます）
    chunk_decoded = chunk.decode()
    chunkjson = None
    has_choices = False
    has_content = False
    has_role = False
    try: 
        chunkjson = json.loads(chunk_decoded[6:])
        has_choices = 'choices' in chunkjson
        if has_choices: has_content = "content" in chunkjson['choices'][0]["delta"]
        if has_choices: has_role = "role" in chunkjson['choices'][0]["delta"]
    except: 
        pass
    return chunk_decoded, chunkjson, has_choices, has_content, has_role

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
    watch_dog_patience = 5 # テキストの翻訳, 5秒で設定できます
    headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from .bridge_all import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS); break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'タイムアウトしました，原始文本 ({retry}/{MAX_RETRY}) ……')

    stream_response =  response.iter_lines()
    result = ''
    json_data = None
    while True:
        try: chunk = next(stream_response).decode()
        except StopIteration: 
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode() # テキストの翻訳，1回再試行する？再失敗したらどうしようもない。
        if len(chunk)==0: continue
        if not chunk.startswith('data:'): 
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAIはリクエストを拒否しました:" + error_msg)
            else:
                raise RuntimeError("OpenAIはリクエストを拒否しました：" + error_msg)
        if ('data: [DONE]' in chunk): break # api2dが正常に完了しました
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0: break
        if "role" in delta: continue
        if "content" in delta:
            result += delta["content"]
            if not console_slience: print(delta["content"], end='')
            if observe_window is not None: 
                # 観測ウィンドウ，取得したデータを表示します
                if len(observe_window) >= 1:
                    observe_window[0] += delta["content"]
                # ウォッチドッグ，期限を超えて餌を与えない場合，それにより終了します
                if len(observe_window) >= 2:
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("ユーザーがプログラムをキャンセルしました。")
        else: raise RuntimeError("テキストの翻訳："+delta)
    if json_data and json_data['finish_reason'] == 'content_filter':
        raise RuntimeError("質問にしない適切な内容が含まれているため、Azureでフィルタリングされました。")
    if json_data and json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("正常に終了，ただし、トークンがしない足して表示されます，出力がしない完全になる原因，一度の入力テキスト量を削減してください。")
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
    if is_any_api_key(inputs):
        chatbot._cookies['api_key'] = inputs
        chatbot.append(("OpenAIのAPIキーを入力してください", what_keys(inputs)))
        yield from update_ui(chatbot=chatbot, history=history, msg="テキストの翻訳") # 画面をリフレッシュする
        return
    elif not is_any_api_key(chatbot._cookies['api_key']):
        chatbot.append((inputs, "原始文本api_key。\n\nテキストの翻訳：テキストの翻訳，次にEnterキーを押して送信します。\n\n2. 長期的な解決策：在config.py中配置。"))
        yield from update_ui(chatbot=chatbot, history=history, msg="原始文本api_key") # 画面をリフレッシュする
        return

    user_input = inputs
    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    raw_input = inputs
    logging.info(f'[raw_input] {raw_input}')
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history, msg="原始文本") # 画面をリフレッシュする

    # check mis-behavior
    if is_the_upload_folder(user_input):
        chatbot[-1] = (inputs, f"[Local Message] 操作エラーが検出されました！ドキュメントをアップロードした後，**関数プラグインエリア**ボタンをクリックして処理してください，「送信」ボタンまたは「基本機能エリア」ボタンをクリックしないでください。")
        yield from update_ui(chatbot=chatbot, history=history, msg="正常") # 画面をリフレッシュする
        time.sleep(2)

    try:
        headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt, stream)
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
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS);break
        except:
            retry += 1
            chatbot[-1] = ((chatbot[-1][0], timeout_bot_msg))
            retry_msg = f"，原始文本 ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield from update_ui(chatbot=chatbot, history=history, msg="タイムアウトしました"+retry_msg) # 画面をリフレッシュする
            if retry > MAX_RETRY: raise TimeoutError

    gpt_replying_buffer = ""
    
    is_head_of_the_stream = True
    if stream:
        stream_response =  response.iter_lines()
        while True:
            try:
                chunk = next(stream_response)
            except StopIteration:
                # OpenAI公式インターフェース以外でこのようなエラーが発生します，OpenAIとAPI2Dはここを通りません
                chunk_decoded = chunk.decode()
                error_msg = chunk_decoded
                # まず、one-apiにdoneデータパケットがない第三者のバグの可能性を排除します
                if len(gpt_replying_buffer.strip()) > 0 and len(error_msg) == 0: 
                    yield from update_ui(chatbot=chatbot, history=history, msg="欠陥のある非公式のOpenAIインターフェースが検出されました，テキストの翻訳。")
                    break
                # テキストの翻訳，エラーを直接返す
                chatbot, history = handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg)
                yield from update_ui(chatbot=chatbot, history=history, msg="非公式のOpenAI APIインターフェースがエラーを返しました:" + chunk.decode()) # 画面をリフレッシュする
                return
            
            # 事前にいくつかの情報を読み取る （例外を判断するために使用するされます）
            chunk_decoded, chunkjson, has_choices, has_content, has_role = decode_chunk(chunk)

            if is_head_of_the_stream and (r'"object":"error"' not in chunk_decoded) and (r"content" not in chunk_decoded):
                # テキストの翻訳
                is_head_of_the_stream = False; continue
            
            if chunk:
                try:
                    # 前者はAPI2Dの終了条件です，後者はOPENAIの終了条件です
                    if ('data: [DONE]' in chunk_decoded) or (len(chunkjson['choices'][0]["delta"]) == 0):
                        # データフローの終了と判断されました，テキストの翻訳
                        logging.info(f'[response] {gpt_replying_buffer}')
                        break
                    # データフローの本体を処理する
                    status_text = f"finish_reason: {chunkjson['choices'][0].get('finish_reason', 'null')}"
                    # ここで例外が発生した場合，通常、テキストが長すぎる，get_full_errorの出力を参照してください
                    if has_content:
                        # 通常の状況では
                        gpt_replying_buffer = gpt_replying_buffer + chunkjson['choices'][0]["delta"]["content"]
                    elif has_role:
                        # いくつかのサードパーティのインターフェースでこのようなエラーが発生します，互換性を保ちましょう
                        continue
                    else:
                        # 一部のゴミの第三者インターフェースでこのようなエラーが発生します
                        gpt_replying_buffer = gpt_replying_buffer + chunkjson['choices'][0]["delta"]["content"]

                    history[-1] = gpt_replying_buffer
                    chatbot[-1] = (history[-2], history[-1])
                    yield from update_ui(chatbot=chatbot, history=history, msg=status_text) # 画面をリフレッシュする
                except Exception as e:
                    yield from update_ui(chatbot=chatbot, history=history, msg="Jsonの解析が通常と異なります") # 画面をリフレッシュする
                    chunk = get_full_error(chunk, stream_response)
                    chunk_decoded = chunk.decode()
                    error_msg = chunk_decoded
                    chatbot, history = handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg)
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json例外" + error_msg) # 画面をリフレッシュする
                    print(error_msg)
                    return

def handle_error(inputs, llm_kwargs, chatbot, history, chunk_decoded, error_msg):
    from .bridge_all import model_info
    openai_website = ' 詳細については、OpenAIにログインしてください https://platform.openai.com/signup'
    if "reduce the length" in error_msg:
        if len(history) >= 2: history[-1] = ""; history[-2] = "" # 現在のオーバーフロー入力をクリアする：history[-2] 是本次输入, history[-1] 今回の出力です
        history = clip_history(inputs=inputs, history=history, tokenizer=model_info[llm_kwargs['llm_model']]['tokenizer'], 
                                               max_token_limit=(model_info[llm_kwargs['llm_model']]['max_token'])) # 原始文本
        chatbot[-1] = (chatbot[-1][0], "[Local Message] 長さを短くしてください。入力が長すぎます, テキストの翻訳, もう一度お試しください。 (もう一度失敗した場合、入力が長すぎる可能性が高いです)")
    elif "does not exist" in error_msg:
        chatbot[-1] = (chatbot[-1][0], f"[Local Message] Model {llm_kwargs['llm_model']} 存在しません。モデルが存在しません, 或者您没有获得体验资格.")
    elif "Incorrect API key" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] しない正なAPIキーです。OpenAIは正しくないAPI_KEYを提供しています, サービスを拒否する. " + openai_website)
    elif "exceeded your current quota" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] 現在のクォータを超えました。OpenAIはアカウントのクォータしない足を理由にしています, サービスを拒否する." + openai_website)
    elif "account is not active" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] アカウントがアクティブではありません。OpenAIはアカウントの無効化を理由にしています, サービスを拒否する." + openai_website)
    elif "associated with a deactivated account" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] 無効化されたアカウントに関連付けられています。OpenAIはアカウントの無効化を理由にしています, サービスを拒否する." + openai_website)
    elif "bad forward key" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] しない正なフォワードキー。API2Dアカウントの残高がしない足しています.")
    elif "Not enough point" in error_msg:
        chatbot[-1] = (chatbot[-1][0], "[Local Message] 原始文本.")
    else:
        from toolbox import regular_txt_to_markdown
        tb_str = '```\n' + trimmed_format_exc() + '```'
        chatbot[-1] = (chatbot[-1][0], f"[Local Message] 例外 \n\n{tb_str} \n\n{regular_txt_to_markdown(chunk_decoded)}")
    return chatbot, history

def generate_payload(inputs, llm_kwargs, history, system_prompt, stream):
    """
    すべての情報を統合する，选择LLM模型，HTTPリクエストを生成する，リクエストの送信の準備をする
    """
    if not is_any_api_key(llm_kwargs['api_key']):
        raise AssertionError("間違ったAPI_KEYを提供しました。\n\nテキストの翻訳：テキストの翻訳，次にEnterキーを押して送信します。\n\n2. 長期的な解決策：在config.py中配置。")

    api_key = select_api_key(llm_kwargs['api_key'], llm_kwargs['llm_model'])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    if API_ORG.startswith('org-'): headers.update({"OpenAI-Organization": API_ORG})
    if llm_kwargs['llm_model'].startswith('azure-'): headers.update({"api-key": api_key})

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
    model = llm_kwargs['llm_model'].strip('api2d-')
    if model == "gpt-3.5-random": # ランダムに選択する, openaiのアクセス頻度制限を回避する
        model = random.choice([
            "gpt-3.5-turbo", 
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-0301",
        ])
        logging.info("Random select model:" + model)

    payload = {
        "model": model,
        "messages": messages, 
        "temperature": llm_kwargs['temperature'],  # 1.0,
        "top_p": llm_kwargs['top_p'],  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    try:
        print(f" {llm_kwargs['llm_model']} : {conversation_cnt} : {inputs[:100]} ..........")
    except:
        print('入力には文字化けが含まれる可能性があります。')
    return headers,payload


