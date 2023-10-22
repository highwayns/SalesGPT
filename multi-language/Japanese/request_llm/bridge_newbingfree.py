"""
========================================================================
原始文本：原始文本
https://github.com/acheong08/EdgeGPT
========================================================================
"""
from .edge_gpt_free import Chatbot as NewbingChatbot
load_message = "待機中NewBing响应。"

"""
========================================================================
第2部分：子プロセスのWorker（呼び出し主体）
========================================================================
"""
import time
import json
import re
import logging
import asyncio
import importlib
import threading
from toolbox import update_ui, get_conf, trimmed_format_exc
from multiprocessing import Process, Pipe

def preprocess_newbing_out(s):
    pattern = r'\^(\d+)\^' # ^数字^に一致する
    sub = lambda m: '('+m.group(1)+')' # 一致した数字を置換値として使用するする
    result = re.sub(pattern, sub, s) # テキストの翻訳
    if '[1]' in result:
        result += '\n\n```reference\n' + "\n".join([r for r in result.split('\n') if r.startswith('[')]) + '\n```\n'
    return result

def preprocess_newbing_out_simple(result):
    if '[1]' in result:
        result += '\n\n```reference\n' + "\n".join([r for r in result.split('\n') if r.startswith('[')]) + '\n```\n'
    return result

class NewBingHandle(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.newbing_model = None
        self.info = ""
        self.success = True
        self.local_history = []
        self.check_dependency()
        self.start()
        self.threadLock = threading.Lock()
        
    def check_dependency(self):
        try:
            self.success = False
            import certifi, httpx, rich
            self.info = "依存関係の検出に合格しました，待機中NewBing响应。現在、複数の人が同時にNewBingインターフェースを呼び出すことはできません（スレッドロックあり），さもなければ、各人のNewBingの問い合わせ履歴が相互に浸透する。NewBingを呼び出すときに，自動的に設定されたプロキシを使用するします。"
            self.success = True
        except:
            self.info = "しない足している依存関係，テキストの翻訳，基本的なpipの依存関係以外，您还需要运行`pip install -r request_llm/requirements_newbing.txt`Newbingの依存関係をインストールする。"
            self.success = False

    def ready(self):
        return self.newbing_model is not None

    async def async_run(self):
        # 設定の読み取り
        NEWBING_STYLE, = get_conf('NEWBING_STYLE')
        from request_llm.bridge_all import model_info
        endpoint = model_info['newbing']['endpoint']
        while True:
            # 待機中
            kwargs = self.child.recv()
            question=kwargs['query']
            history=kwargs['history']
            system_prompt=kwargs['system_prompt']

            # 原始文本
            if len(self.local_history) > 0 and len(history)==0:
                await self.newbing_model.reset()
                self.local_history = []

            # 質問を始める
            prompt = ""
            if system_prompt not in self.local_history:
                self.local_history.append(system_prompt)
                prompt += system_prompt + '\n'

            # テキストの翻訳
            for ab in history:
                a, b = ab
                if a not in self.local_history:
                    self.local_history.append(a)
                    prompt += a + '\n'

            # 問題
            prompt += question
            self.local_history.append(question)
            print('question:', prompt)
            # 送信
            async for final, response in self.newbing_model.ask_stream(
                prompt=question,
                conversation_style=NEWBING_STYLE,     # ["creative", "balanced", "precise"]
                wss_link=endpoint,                    # "wss://sydney.bing.com/sydney/ChatHub"
            ):
                if not final:
                    print(response)
                    self.child.send(str(response))
                else:
                    print('-------- receive final ---------')
                    self.child.send('[Finish]')
                    # self.local_history.append(response)

    
    def run(self):
        """
        この関数はサブプロセスで実行されます
        """
        # 初回実行，原始文本
        self.success = False
        self.local_history = []
        if (self.newbing_model is None) or (not self.success):
            # プロキシの設定
            proxies, NEWBING_COOKIES = get_conf('proxies', 'NEWBING_COOKIES')
            if proxies is None: 
                self.proxies_https = None
            else: 
                self.proxies_https = proxies['https']

            if (NEWBING_COOKIES is not None) and len(NEWBING_COOKIES) > 100:
                try:
                    cookies = json.loads(NEWBING_COOKIES)
                except:
                    self.success = False
                    tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
                    self.child.send(f'[Local Message] NEWBING_COOKIESが入力されていないか、形式が正しくありません。')
                    self.child.send('[Fail]'); self.child.send('[Finish]')
                    raise RuntimeError(f"NEWBING_COOKIESが入力されていないか、形式が正しくありません。")
            else:
                cookies = None

            try:
                self.newbing_model = NewbingChatbot(proxy=self.proxies_https, cookies=cookies)
            except:
                self.success = False
                tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
                self.child.send(f'[Local Message] Newbingコンポーネントをロードできません。{tb_str}')
                self.child.send('[Fail]')
                self.child.send('[Finish]')
                raise RuntimeError(f"Newbingコンポーネントをロードできません。")

        self.success = True
        try:
            # タスクの待機状態に入る
            asyncio.run(self.async_run())
        except Exception:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            self.child.send(f'[Local Message] Newbing 请求失敗，テキストの翻訳，プロキシプロトコルの変更をお勧めします（原始文本）またはプロキシノード {tb_str}.')
            self.child.send('[Fail]')
            self.child.send('[Finish]')
        
    def stream_chat(self, **kwargs):
        """
        この関数はメインプロセスで実行されます
        """
        self.threadLock.acquire()   # 获取テキストの翻訳锁
        self.parent.send(kwargs)    # 原始文本
        while True:
            res = self.parent.recv()                            # newbingの返信を待つフラグメント
            if res == '[Finish]': break                         # 終了
            elif res == '[Fail]': self.success = False; break   # 失敗
            else: yield res                                     # newbingの返信のフラグメント
        self.threadLock.release()   # スレッドロックを解放する


"""
========================================================================
テキストの翻訳：メインプロセスは統一された関数インターフェースを呼び出します
========================================================================
"""
global newbingfree_handle
newbingfree_handle = None

def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        マルチスレッドのテキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    global newbingfree_handle
    if (newbingfree_handle is None) or (not newbingfree_handle.success):
        newbingfree_handle = NewBingHandle()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + newbingfree_handle.info
        if not newbingfree_handle.success: 
            error = newbingfree_handle.info
            newbingfree_handle = None
            raise RuntimeError(error)

    # sys_promptインターフェースはありません，したがって、promptを履歴に追加する
    history_feedin = []
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    watch_dog_patience = 5 # ウォッチドッグ (watchdog) の忍耐力, 5秒で設定できます
    response = ""
    if len(observe_window) >= 1: observe_window[0] = "[Local Message]: NewBingの応テキストの翻訳を待っています ..."
    for response in newbingfree_handle.stream_chat(query=inputs, history=history_feedin, system_prompt=sys_prompt, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        if len(observe_window) >= 1:  observe_window[0] = preprocess_newbing_out_simple(response)
        if len(observe_window) >= 2:  
            if (time.time()-observe_window[1]) > watch_dog_patience:
                raise RuntimeError("原始文本。")
    return preprocess_newbing_out_simple(response)

def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        テキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    chatbot.append((inputs, "[Local Message]: NewBingの応テキストの翻訳を待っています ..."))

    global newbingfree_handle
    if (newbingfree_handle is None) or (not newbingfree_handle.success):
        newbingfree_handle = NewBingHandle()
        chatbot[-1] = (inputs, load_message + "\n\n" + newbingfree_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not newbingfree_handle.success: 
            newbingfree_handle = None
            return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    history_feedin = []
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    chatbot[-1] = (inputs, "[Local Message]: NewBingの応テキストの翻訳を待っています ...")
    response = "[Local Message]: NewBingの応テキストの翻訳を待っています ..."
    yield from update_ui(chatbot=chatbot, history=history, msg="NewBingの応テキストの翻訳が遅い，原始文本，原始文本。")
    for response in newbingfree_handle.stream_chat(query=inputs, history=history_feedin, system_prompt=system_prompt, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        chatbot[-1] = (inputs, preprocess_newbing_out(response))
        yield from update_ui(chatbot=chatbot, history=history, msg="NewBingの応テキストの翻訳が遅い，原始文本，原始文本。")
    if response == "[Local Message]: NewBingの応テキストの翻訳を待っています ...": response = "[Local Message]: NewBingの応テキストの翻訳が異常です，画面を更新して再試行してください ..."
    history.extend([inputs, response])
    logging.info(f'[raw_input] {inputs}')
    logging.info(f'[response] {response}')
    yield from update_ui(chatbot=chatbot, history=history, msg="すべての応テキストの翻訳を完了する，テキストの翻訳。")

