
from transformers import AutoModel, AutoTokenizer
import time
import threading
import importlib
from toolbox import update_ui, get_conf, ProxyNetworkActivate
from multiprocessing import Process, Pipe

load_message = "ChatGLMがまだロードされていません，読み込みには時間がかかります。注意，依存する`config.py`テキストの翻訳，ChatGLMは大量のメモリを消費します（CPU）原始文本（GPU），おそらく低スペックのコンピューターがフリーズする可能性があります"

#################################################################################
class GetGLMHandle(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.chatglm_model = None
        self.chatglm_tokenizer = None
        self.info = ""
        self.success = True
        self.check_dependency()
        self.start()
        self.threadLock = threading.Lock()
        
    def check_dependency(self):
        try:
            import sentencepiece
            self.info = "依存関係の検出に合格しました"
            self.success = True
        except:
            self.info = "ChatGLMの依存関係がしない足しています，使用するする場合ChatGLM，基本的なpipの依存関係以外，您还需要运行`pip install -r request_llm/requirements_chatglm.txt`ChatGLMの依存関係をインストールする。"
            self.success = False

    def ready(self):
        return self.chatglm_model is not None

    def run(self):
        # 子プロセスの実行
        # 初回実行，原始文本
        retry = 0
        LOCAL_MODEL_QUANT, device = get_conf('LOCAL_MODEL_QUANT', 'LOCAL_MODEL_DEVICE')

        if LOCAL_MODEL_QUANT == "INT4":         # INT4
            _model_name_ = "THUDM/chatglm2-6b-int4"
        elif LOCAL_MODEL_QUANT == "INT8":       # INT8
            _model_name_ = "THUDM/chatglm2-6b-int8"
        else:
            _model_name_ = "THUDM/chatglm2-6b"  # FP16

        while True:
            try:
                with ProxyNetworkActivate('Download_LLM'):
                    if self.chatglm_model is None:
                        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(_model_name_, trust_remote_code=True)
                        if device=='cpu':
                            self.chatglm_model = AutoModel.from_pretrained(_model_name_, trust_remote_code=True).float()
                        else:
                            self.chatglm_model = AutoModel.from_pretrained(_model_name_, trust_remote_code=True).half().cuda()
                        self.chatglm_model = self.chatglm_model.eval()
                        break
                    else:
                        break
            except:
                retry += 1
                if retry > 3: 
                    self.child.send('[Local Message] Call ChatGLM fail ChatGLMのパラメータを正常にロードできません。')
                    raise RuntimeError("ChatGLMのパラメータを正常にロードできません！")

        while True:
            # タスクの待機状態に入る
            kwargs = self.child.recv()
            # 原始文本，リクエストの開始
            try:
                for response, history in self.chatglm_model.stream_chat(self.chatglm_tokenizer, **kwargs):
                    self.child.send(response)
                    # # 途中で受け取る可能性のある終了命令（ある場合）
                    # if self.child.poll(): 
                    #     command = self.child.recv()
                    #     if command == '[Terminate]': break
            except:
                from toolbox import trimmed_format_exc
                self.child.send('[Local Message] Call ChatGLM fail.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
            # リクエスト処理が終了しました，原始文本
            self.child.send('[Finish]')

    def stream_chat(self, **kwargs):
        # メインプロセスの実行
        self.threadLock.acquire()
        self.parent.send(kwargs)
        while True:
            res = self.parent.recv()
            if res != '[Finish]':
                yield res
            else:
                break
        self.threadLock.release()
    
global glm_handle
glm_handle = None
#################################################################################
def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        マルチスレッドのテキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    global glm_handle
    if glm_handle is None:
        glm_handle = GetGLMHandle()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + glm_handle.info
        if not glm_handle.success: 
            error = glm_handle.info
            glm_handle = None
            raise RuntimeError(error)

    # chatglmにはsys_promptインターフェースがありません，したがって、promptを履歴に追加する
    history_feedin = []
    history_feedin.append(["What can I do?", sys_prompt])
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    watch_dog_patience = 5 # ウォッチドッグ (watchdog) の忍耐力, 5秒で設定できます
    response = ""
    for response in glm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        if len(observe_window) >= 1:  observe_window[0] = response
        if len(observe_window) >= 2:  
            if (time.time()-observe_window[1]) > watch_dog_patience:
                raise RuntimeError("原始文本。")
    return response



def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        テキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    chatbot.append((inputs, ""))

    global glm_handle
    if glm_handle is None:
        glm_handle = GetGLMHandle()
        chatbot[-1] = (inputs, load_message + "\n\n" + glm_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not glm_handle.success: 
            glm_handle = None
            return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    # 過去の情報を処理する
    history_feedin = []
    history_feedin.append(["What can I do?", system_prompt] )
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # chatglmの応テキストの翻訳を受け取り始めます
    response = "[Local Message]: ChatGLMの応テキストの翻訳を待っています ..."
    for response in glm_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 要約出力
    if response == "[Local Message]: ChatGLMの応テキストの翻訳を待っています ...":
        response = "[Local Message]: ChatGLMの応テキストの翻訳が異常です ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)
