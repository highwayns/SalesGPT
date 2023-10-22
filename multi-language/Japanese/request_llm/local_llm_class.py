from transformers import AutoModel, AutoTokenizer
import time
import threading
import importlib
from toolbox import update_ui, get_conf, Singleton
from multiprocessing import Process, Pipe

def SingletonLocalLLM(cls):
    """
    単一のインスタンスデコレータ
    """
    _instance = {}
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
            return _instance[cls]
        elif _instance[cls].corrupted:
            _instance[cls] = cls(*args, **kargs)
            return _instance[cls]
        else:
            return _instance[cls]
    return _singleton

class LocalLLMHandle(Process):
    def __init__(self):
        # ⭐メインプロセスを実行する
        super().__init__(daemon=True)
        self.corrupted = False
        self.load_model_info()
        self.parent, self.child = Pipe()
        self.running = True
        self._model = None
        self._tokenizer = None
        self.info = ""
        self.check_dependency()
        self.start()
        self.threadLock = threading.Lock()

    def load_model_info(self):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子プロセスの実行
        raise NotImplementedError("Method not implemented yet")
        self.model_name = ""
        self.cmd_to_install = ""

    def load_model_and_tokenizer(self):
        """
        This function should return the model and the tokenizer
        """
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子プロセスの実行
        raise NotImplementedError("Method not implemented yet")

    def llm_stream_generator(self, **kwargs):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子プロセスの実行
        raise NotImplementedError("Method not implemented yet")
        
    def try_to_import_special_deps(self, **kwargs):
        """
        import something that will raise error if the user does not install requirement_*.txt
        """
        # ⭐メインプロセスを実行する
        raise NotImplementedError("Method not implemented yet")

    def check_dependency(self):
        # ⭐メインプロセスを実行する
        try:
            self.try_to_import_special_deps()
            self.info = "依存関係の検出に合格しました"
            self.running = True
        except:
            self.info = f"原始文本{self.model_name}の依存関係，使用するする場合{self.model_name}，基本的なpipの依存関係以外，您还需要运行{self.cmd_to_install}インストール{self.model_name}の依存関係。"
            self.running = False

    def run(self):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子プロセスの実行
        # 初回実行，原始文本
        try:
            self._model, self._tokenizer = self.load_model_and_tokenizer()
        except:
            self.running = False
            from toolbox import trimmed_format_exc
            self.child.send(f'[Local Message] しない能正常加载{self.model_name}原始文本.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
            self.child.send('[FinishBad]')
            raise RuntimeError(f"しない能正常加载{self.model_name}原始文本！")

        while True:
            # タスクの待機状態に入る
            kwargs = self.child.recv()
            # 原始文本，リクエストの開始
            try:
                for response_full in self.llm_stream_generator(**kwargs):
                    self.child.send(response_full)
                self.child.send('[Finish]')
                # リクエスト処理が終了しました，原始文本
            except:
                from toolbox import trimmed_format_exc
                self.child.send(f'[Local Message] 呼び出し{self.model_name}失敗.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
                self.child.send('[Finish]')

    def stream_chat(self, **kwargs):
        # ⭐メインプロセスを実行する
        self.threadLock.acquire()
        self.parent.send(kwargs)
        while True:
            res = self.parent.recv()
            if res == '[Finish]': 
                break
            if res == '[FinishBad]': 
                self.running = False
                self.corrupted = True
                break
            else: 
                yield res
        self.threadLock.release()
    


def get_local_llm_predict_fns(LLMSingletonClass, model_name):
    load_message = f"{model_name}まだ読み込まれていません，読み込みには時間がかかります。注意，依存する`config.py`テキストの翻訳，{model_name}大量のメモリを消費する（CPU）原始文本（GPU），おそらく低スペックのコンピューターがフリーズする可能性があります"

    def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
        """
            ⭐マルチスレッドメソッド
            関数の説明については、request_llm/bridge_all.pyを参照してください
        """
        _llm_handle = LLMSingletonClass()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + _llm_handle.info
        if not _llm_handle.running: raise RuntimeError(_llm_handle.info)

        # chatglmにはsys_promptインターフェースがありません，したがって、promptを履歴に追加する
        history_feedin = []
        history_feedin.append([sys_prompt, "Certainly!"])
        for i in range(len(history)//2):
            history_feedin.append([history[2*i], history[2*i+1]] )

        watch_dog_patience = 5 # ウォッチドッグ (watchdog) の忍耐力, 5秒で設定できます
        response = ""
        for response in _llm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
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

        _llm_handle = LLMSingletonClass()
        chatbot[-1] = (inputs, load_message + "\n\n" + _llm_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not _llm_handle.running: raise RuntimeError(_llm_handle.info)

        if additional_fn is not None:
            from core_functional import handle_core_functionality
            inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

        # 過去の情報を処理する
        history_feedin = []
        history_feedin.append([system_prompt, "Certainly!"])
        for i in range(len(history)//2):
            history_feedin.append([history[2*i], history[2*i+1]] )

        # テキストの翻訳
        response = f"[Local Message]: 待機中{model_name}原始文本 ..."
        for response in _llm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
            chatbot[-1] = (inputs, response)
            yield from update_ui(chatbot=chatbot, history=history)

        # 要約出力
        if response == f"[Local Message]: 待機中{model_name}原始文本 ...":
            response = f"[Local Message]: {model_name}応テキストの翻訳異常 ..."
        history.extend([inputs, response])
        yield from update_ui(chatbot=chatbot, history=history)

    return predict_no_ui_long_connection, predict