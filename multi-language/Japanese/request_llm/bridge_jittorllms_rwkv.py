
from transformers import AutoModel, AutoTokenizer
import time
import threading
import importlib
from toolbox import update_ui, get_conf
from multiprocessing import Process, Pipe

load_message = "jittorllmsがまだロードされていません，読み込みには時間がかかります。注意，原始文本，それ以外の場合、メモリオーバーフローが発生し、フリーズが発生する可能性があります，依存する`config.py`テキストの翻訳，jittorllmsは大量のメモリを消費します（CPU）原始文本（GPU），おそらく低スペックのコンピューターがフリーズする可能性があります"

#################################################################################
class GetGLMHandle(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.jittorllms_model = None
        self.info = ""
        self.local_history = []
        self.success = True
        self.check_dependency()
        self.start()
        self.threadLock = threading.Lock()
        
    def check_dependency(self):
        try:
            import pandas
            self.info = "依存関係の検出に合格しました"
            self.success = True
        except:
            from toolbox import trimmed_format_exc
            self.info = r"jittorllmsの依存関係がしない足しています，jittorllmsを使用するする場合，基本的なpipの依存関係以外，您还需要运行`pip install -r request_llm/requirements_jittorllms.txt -i https://pypi.jittor.org/simple -I`"+\
                        r"と`git clone https://gitlink.org.cn/jittor/JittorLLMs.git --depth 1 request_llm/jittorllms`jittorllmsの依存関係をインストールするための2つのコマンド（在项目根目录运行这两テキストの翻訳指令）。" +\
                        r"警告：jittorllmsの依存関係をインストールすると、既存のpytorch環境が完全に破壊されます，Docker環境の使用するをお勧めします！" + trimmed_format_exc()
            self.success = False

    def ready(self):
        return self.jittorllms_model is not None

    def run(self):
        # 子プロセスの実行
        # 初回実行，原始文本
        def validate_path():
            import os, sys
            dir_name = os.path.dirname(__file__)
            env = os.environ.get("PATH", "")
            os.environ["PATH"] = env.replace('/cuda/bin', '/x/bin')
            root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
            os.chdir(root_dir_assume + '/request_llm/jittorllms')
            sys.path.append(root_dir_assume + '/request_llm/jittorllms')
        validate_path() # validate path so you can run from base directory

        def load_model():
            import types
            try:
                if self.jittorllms_model is None:
                    device, = get_conf('LOCAL_MODEL_DEVICE')
                    from .jittorllms.models import get_model
                    # availabel_models = ["chatglm", "pangualpha", "llama", "chatrwkv"]
                    args_dict = {'model': 'chatrwkv'}
                    print('self.jittorllms_model = get_model(types.SimpleNamespace(**args_dict))')
                    self.jittorllms_model = get_model(types.SimpleNamespace(**args_dict))
                    print('done get model')
            except:
                self.child.send('[Local Message] jittorllmsのパラメータを正常にロードできません。')
                raise RuntimeError("テキストの翻訳")
        print('load_model')
        load_model()

        # タスクの待機状態に入る
        print('タスクの待機状態に入る')
        while True:
            # タスクの待機状態に入る
            kwargs = self.child.recv()
            query = kwargs['query']
            history = kwargs['history']
            # 原始文本
            if len(self.local_history) > 0 and len(history)==0:
                print('リセットをトリガーする')
                self.jittorllms_model.reset()
            self.local_history.append(query)

            print('原始文本，リクエストの開始')
            try:
                for response in self.jittorllms_model.stream_chat(query, history):
                    print(response)
                    self.child.send(response)
            except:
                from toolbox import trimmed_format_exc
                print(trimmed_format_exc())
                self.child.send('[Local Message] Call jittorllms fail.')
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
    
global rwkv_glm_handle
rwkv_glm_handle = None
#################################################################################
def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        マルチスレッドのテキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    global rwkv_glm_handle
    if rwkv_glm_handle is None:
        rwkv_glm_handle = GetGLMHandle()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + rwkv_glm_handle.info
        if not rwkv_glm_handle.success: 
            error = rwkv_glm_handle.info
            rwkv_glm_handle = None
            raise RuntimeError(error)

    # jittorllmsにはsys_promptインターフェースがありません，したがって、promptを履歴に追加する
    history_feedin = []
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    watch_dog_patience = 5 # ウォッチドッグ (watchdog) の忍耐力, 5秒で設定できます
    response = ""
    for response in rwkv_glm_handle.stream_chat(query=inputs, history=history_feedin, system_prompt=sys_prompt, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        print(response)
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

    global rwkv_glm_handle
    if rwkv_glm_handle is None:
        rwkv_glm_handle = GetGLMHandle()
        chatbot[-1] = (inputs, load_message + "\n\n" + rwkv_glm_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not rwkv_glm_handle.success: 
            rwkv_glm_handle = None
            return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    # 過去の情報を処理する
    history_feedin = []
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # jittorllmsの応テキストの翻訳を受け取り始めます
    response = "[Local Message]: jittorllmsの応テキストの翻訳を待っています ..."
    for response in rwkv_glm_handle.stream_chat(query=inputs, history=history_feedin, system_prompt=system_prompt, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 要約出力
    if response == "[Local Message]: jittorllmsの応テキストの翻訳を待っています ...":
        response = "[Local Message]: jittorllmsの応テキストの翻訳が異常です ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)
