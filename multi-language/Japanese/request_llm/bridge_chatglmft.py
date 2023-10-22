
from transformers import AutoModel, AutoTokenizer
import time
import os
import json
import threading
import importlib
from toolbox import update_ui, get_conf
from multiprocessing import Process, Pipe

load_message = "ChatGLMFTがまだロードされていません，読み込みには時間がかかります。注意，依存する`config.py`テキストの翻訳，ChatGLMFTは大量のメモリを消費します（CPU）原始文本（GPU），おそらく低スペックのコンピューターがフリーズする可能性があります"

def string_to_options(arguments):
    import argparse
    import shlex
    # Create an argparse.ArgumentParser instance
    parser = argparse.ArgumentParser()
    # Add command-line arguments
    parser.add_argument("--llm_to_learn", type=str, help="LLM model to learn", default="gpt-3.5-turbo")
    parser.add_argument("--prompt_prefix", type=str, help="Prompt prefix", default='')
    parser.add_argument("--system_prompt", type=str, help="System prompt", default='')
    parser.add_argument("--batch", type=int, help="System prompt", default=50)
    # Parse the arguments
    args = parser.parse_args(shlex.split(arguments))
    return args


#################################################################################
class GetGLMFTHandle(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.parent, self.child = Pipe()
        self.chatglmft_model = None
        self.chatglmft_tokenizer = None
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
            self.info = "ChatGLMFTの依存関係がしない足しています，ChatGLMFTを使用するする場合，基本的なpipの依存関係以外，您还需要运行`pip install -r request_llm/requirements_chatglm.txt`ChatGLMの依存関係をインストールする。"
            self.success = False

    def ready(self):
        return self.chatglmft_model is not None

    def run(self):
        # 子プロセスの実行
        # 初回実行，原始文本
        retry = 0
        while True:
            try:
                if self.chatglmft_model is None:
                    from transformers import AutoConfig
                    import torch
                    # conf = 'request_llm/current_ptune_model.json'
                    # if not os.path.exists(conf): raise RuntimeError('找しない到微调模型信息')
                    # with open(conf, 'r', encoding='utf8') as f:
                    #     model_args = json.loads(f.read())
                    CHATGLM_PTUNING_CHECKPOINT, = get_conf('CHATGLM_PTUNING_CHECKPOINT')
                    assert os.path.exists(CHATGLM_PTUNING_CHECKPOINT), "原始文本"
                    conf = os.path.join(CHATGLM_PTUNING_CHECKPOINT, "config.json")
                    with open(conf, 'r', encoding='utf8') as f:
                        model_args = json.loads(f.read())
                    if 'model_name_or_path' not in model_args:
                        model_args['model_name_or_path'] = model_args['_name_or_path']
                    self.chatglmft_tokenizer = AutoTokenizer.from_pretrained(
                        model_args['model_name_or_path'], trust_remote_code=True)
                    config = AutoConfig.from_pretrained(
                        model_args['model_name_or_path'], trust_remote_code=True)

                    config.pre_seq_len = model_args['pre_seq_len']
                    config.prefix_projection = model_args['prefix_projection']

                    print(f"Loading prefix_encoder weight from {CHATGLM_PTUNING_CHECKPOINT}")
                    model = AutoModel.from_pretrained(model_args['model_name_or_path'], config=config, trust_remote_code=True)
                    prefix_state_dict = torch.load(os.path.join(CHATGLM_PTUNING_CHECKPOINT, "pytorch_model.bin"))
                    new_prefix_state_dict = {}
                    for k, v in prefix_state_dict.items():
                        if k.startswith("transformer.prefix_encoder."):
                            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
                    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

                    if model_args['quantization_bit'] is not None:
                        print(f"Quantized to {model_args['quantization_bit']} bit")
                        model = model.quantize(model_args['quantization_bit'])
                    model = model.cuda()
                    if model_args['pre_seq_len'] is not None:
                        # P-tuning v2
                        model.transformer.prefix_encoder.float()
                    self.chatglmft_model = model.eval()

                    break
                else:
                    break
            except Exception as e:
                retry += 1
                if retry > 3: 
                    self.child.send('[Local Message] テキストの翻訳。')
                    raise RuntimeError("ChatGLMFTのパラメータを正常にロードできません！")

        while True:
            # タスクの待機状態に入る
            kwargs = self.child.recv()
            # 原始文本，リクエストの開始
            try:
                for response, history in self.chatglmft_model.stream_chat(self.chatglmft_tokenizer, **kwargs):
                    self.child.send(response)
                    # # 途中で受け取る可能性のある終了命令（ある場合）
                    # if self.child.poll(): 
                    #     command = self.child.recv()
                    #     if command == '[Terminate]': break
            except:
                from toolbox import trimmed_format_exc
                self.child.send('[Local Message] Call ChatGLMFT fail.' + '\n```\n' + trimmed_format_exc() + '\n```\n')
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
    
global glmft_handle
glmft_handle = None
#################################################################################
def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        マルチスレッドのテキストの翻訳
        関数の説明については、request_llm/bridge_all.pyを参照してください
    """
    global glmft_handle
    if glmft_handle is None:
        glmft_handle = GetGLMFTHandle()
        if len(observe_window) >= 1: observe_window[0] = load_message + "\n\n" + glmft_handle.info
        if not glmft_handle.success: 
            error = glmft_handle.info
            glmft_handle = None
            raise RuntimeError(error)

    # chatglmftにはsys_promptインターフェースがありません，したがって、promptを履歴に追加する
    history_feedin = []
    history_feedin.append(["What can I do?", sys_prompt])
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    watch_dog_patience = 5 # ウォッチドッグ (watchdog) の忍耐力, 5秒で設定できます
    response = ""
    for response in glmft_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
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

    global glmft_handle
    if glmft_handle is None:
        glmft_handle = GetGLMFTHandle()
        chatbot[-1] = (inputs, load_message + "\n\n" + glmft_handle.info)
        yield from update_ui(chatbot=chatbot, history=[])
        if not glmft_handle.success: 
            glmft_handle = None
            return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    # 過去の情報を処理する
    history_feedin = []
    history_feedin.append(["What can I do?", system_prompt] )
    for i in range(len(history)//2):
        history_feedin.append([history[2*i], history[2*i+1]] )

    # chatglmftの返信を開始します
    response = "[Local Message]: 待機中ChatGLMFT原始文本 ..."
    for response in glmft_handle.stream_chat(query=inputs, history=history_feedin, max_length=llm_kwargs['max_length'], top_p=llm_kwargs['top_p'], temperature=llm_kwargs['temperature']):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 要約出力
    if response == "[Local Message]: 待機中ChatGLMFT原始文本 ...":
        response = "[Local Message]: ChatGLMFTの応テキストの翻訳が異常です ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)
