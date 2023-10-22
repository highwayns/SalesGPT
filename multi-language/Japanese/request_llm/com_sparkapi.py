from toolbox import get_conf
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import threading, time

timeout_bot_msg = '[Local Message] Request timeout. Network error.'

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # URLを生成する
    def create_url(self):
        # 原始文本
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 文字列を連結する
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # hmac-sha256で暗号化する
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # リクエストの認証パラメータを辞書に組み合わせる
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 認証パラメータを連結する，URLを生成する
        url = self.gpt_url + '?' + urlencode(v)
        # ここには接続時に表示されるURLが印刷されます,このデモを参考にする際には、上記のコメントを解除してください，比对相同テキストの翻訳時生成的url与自己代码生成的url是否一致
        return url



class SparkRequestInstance():
    def __init__(self):
        XFYUN_APPID, XFYUN_API_SECRET, XFYUN_API_KEY = get_conf('XFYUN_APPID', 'XFYUN_API_SECRET', 'XFYUN_API_KEY')
        if XFYUN_APPID == '00000000' or XFYUN_APPID == '': raise RuntimeError('请配置讯飞星火大模型的XFYUN_APPID, XFYUN_API_KEY, XFYUN_API_SECRET')
        self.appid = XFYUN_APPID
        self.api_secret = XFYUN_API_SECRET
        self.api_key = XFYUN_API_KEY
        self.gpt_url = "ws://spark-api.xf-yun.com/v1.1/chat"
        self.gpt_url_v2 = "ws://spark-api.xf-yun.com/v2.1/chat"

        self.time_to_yield_event = threading.Event()
        self.time_to_exit_event = threading.Event()

        self.result_buf = ""

    def generate(self, inputs, llm_kwargs, history, system_prompt):
        llm_kwargs = llm_kwargs
        history = history
        system_prompt = system_prompt
        import _thread as thread
        thread.start_new_thread(self.create_blocking_request, (inputs, llm_kwargs, history, system_prompt))
        while True:
            self.time_to_yield_event.wait(timeout=1)
            if self.time_to_yield_event.is_set():
                yield self.result_buf
            if self.time_to_exit_event.is_set():
                return self.result_buf


    def create_blocking_request(self, inputs, llm_kwargs, history, system_prompt):
        if llm_kwargs['llm_model'] == 'sparkv2':
            gpt_url = self.gpt_url_v2
        else:
            gpt_url = self.gpt_url

        wsParam = Ws_Param(self.appid, self.api_key, self.api_secret, gpt_url)
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()

        # WebSocket接続の確立を受け取る処理
        def on_open(ws):
            import _thread as thread
            thread.start_new_thread(run, (ws,))

        def run(ws, *args):
            data = json.dumps(gen_params(ws.appid, *ws.all_args))
            ws.send(data)

        # ウェブソケットメッセージを受信しました
        def on_message(ws, message):
            data = json.loads(message)
            code = data['header']['code']
            if code != 0:
                print(f'リクエストエラー: {code}, {data}')
                self.result_buf += str(data)
                ws.close()
                self.time_to_exit_event.set()
            else:
                choices = data["payload"]["choices"]
                status = choices["status"]
                content = choices["text"][0]["content"]
                ws.content += content
                self.result_buf += content
                if status == 2:
                    ws.close()
                    self.time_to_exit_event.set()
            self.time_to_yield_event.set()

        # WebSocketエラーの処理
        def on_error(ws, error):
            print("error:", error)
            self.time_to_exit_event.set()

        # ウェブソケットのクローズ処理を受信しました
        def on_close(ws, *args):
            self.time_to_exit_event.set()

        # websocket
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
        ws.appid = self.appid
        ws.content = ""
        ws.all_args = (inputs, llm_kwargs, history, system_prompt)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

def generate_message_payload(inputs, llm_kwargs, history, system_prompt):
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
    return messages


def gen_params(appid, inputs, llm_kwargs, history, system_prompt):
    """
    appidとユーザーの質問に基づいてリクエストパラメータを生成する
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": "generalv2" if llm_kwargs['llm_model'] == 'sparkv2' else "general",
                "temperature": llm_kwargs["temperature"],
                "random_threshold": 0.5,
                "max_tokens": 4096,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": generate_message_payload(inputs, llm_kwargs, history, system_prompt)
            }
        }
    }
    return data

