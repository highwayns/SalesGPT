"""
    以下のすべての設定は環境変数を上書きすることもサポートしています，環境変数の設定形式はdocker-compose.ymlを参照してください。
    テキストの翻訳：テキストの翻訳 > config_private.py > config.py
    --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    All the following configurations also support using environment variables to override, 
    and the environment variable configuration format can be seen in docker-compose.yml. 
    Configuration reading priority: environment variable > config_private.py > config.py
"""

# [step 1]>> API_KEY = "sk-123456789xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx123456789"。非常にまれなケースでは，テキストの翻訳（org-123456789abcdefghijklmnoのような形式），下にスクロールしてください，找 API_ORG 设置项
API_KEY = "ここにAPIキーを入力してください"    # テキストの翻訳，英文コンマで分割する，例如API_KEY = "sk-openaikey1,sk-openaikey2,fkxxxx-api2dkey3,azure-apikey4"


# [step 2]>> テキストの翻訳，海外サーバーに直接デプロイする場合，ここは変更しないでください；ローカルまたは地域制限のない大規模モデルを使用するする場合，テキストの翻訳
USE_PROXY = False
if USE_PROXY:
    """
    記入形式は [テキストの翻訳]://  [アドレス] :[ポート]，記入する前に、USE_PROXYをTrueに変更するのを忘れないでください，海外サーバーに直接デプロイする場合，テキストの翻訳
            <設定チュートリアル＆ビデオチュートリアル> https://github.com/binary-husky/gpt_academic/issues/1>
    [テキストの翻訳] 一般的なプロトコルはsocks5h/http以外ありません; 例えば v2**y と ss* のデフォルトのローカルプロトコルはsocks5hです; cl**hのデフォルトのローカルプロトコルはhttpです
    [アドレス] テキストの翻訳，テキストの翻訳（テキストの翻訳）
    [ポート] テキストの翻訳。異なるプロキシソフトウェアのインターフェースは異なります，テキストの翻訳
    """
    # プロキシネットワークのアドレス，あなたの*学*ウェブソフトウェアを開いてプロキシのプロトコルを確認してください(socks5h / http)、アドレス(localhost)およびポート(11284)
    proxies = {
        #          [テキストの翻訳]://  [アドレス]  :[ポート]
        "http":  "socks5h://localhost:11284",  # 再例如  "http":  "http://127.0.0.1:7890",
        "https": "socks5h://localhost:11284",  # 再例如  "https": "http://127.0.0.1:7890",
    }
else:
    proxies = None

# ------------------------------------ 原始文本, ただし、ほとんどの場合は変更する必要はありません ------------------------------------

# URLリダイレクトの再設定，API_URLの変更を実現する（テキストの翻訳，テキストの翻訳）
# テキストの翻訳: API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "在这里填写重定向的api.openai.com的URL"} 
# 例を挙げる: API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "https://reverse-proxy-url/v1/chat/completions"}
API_URL_REDIRECT = {}


# マルチスレッド関数プラグイン内で，デフォルトでは、同時にOpenAIにアクセスできるスレッドの数はいくつですか。テキストの翻訳，Pay-as-you-goユーザーの制限は1分間に3500回です
# 一言で言えば：無料（5刀）ユーザーが3を入力します，OpenAIにクレジットカードがバインドされているユーザーは、16またはそれ以上を入力できます。制限を引き上げるには、クエリを参照してください：https://platform.openai.com/docs/guides/rate-limits/overview
DEFAULT_WORKER_NUM = 3


# カラーテーマ, テキストの翻訳 ["Default", "Chuanhu-Small-and-Beautiful", "High-Contrast"]
# その他のトピック, Gradioテーマストアをご覧ください: https://huggingface.co/spaces/gradio/theme-gallery オプション ["Gstaff/Xkcd", "NoCrypt/Miku", ...]
THEME = "Default"
AVAIL_THEMES = ["Default", "Chuanhu-Small-and-Beautiful", "High-Contrast", "Gstaff/Xkcd", "NoCrypt/Miku"]


# 対話ウィンドウの高さ （仅在LAYOUT="TOP-DOWN"時生效）
CHATBOT_HEIGHT = 1115


# コードのハイライト
CODE_HIGHLIGHT = True


# 窗口布局
LAYOUT = "LEFT-RIGHT"   # "LEFT-RIGHT"（テキストの翻訳） # "TOP-DOWN"（上下のレイアウト）


# ダークモード/ライトモード
DARK_MODE = True        


# OpenAIにリクエストを送信した後，タイムアウトと判断するまでの待機時間
TIMEOUT_SECONDS = 30


# ウェブページのポート, テキストの翻訳
WEB_PORT = -1


# テキストの翻訳（ネットワークの遅延、プロキシの失敗、KEYの無効化），リトライ回数の制限
MAX_RETRY = 2


# テキストの翻訳
DEFAULT_FN_GROUPS = ['対話', '编程', '学術', '原始文本']


# モデルの選択は (注意: LLM_MODELはデフォルトで選択されたモデルです, AVAIL_LLM_MODELSリストに含まれている必要があります )
LLM_MODEL = "gpt-3.5-turbo" # オプション ↓↓↓
AVAIL_LLM_MODELS = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "azure-gpt-3.5", "api2d-gpt-3.5-turbo",
                    "gpt-4", "gpt-4-32k", "azure-gpt-4", "api2d-gpt-4", "chatglm", "moss", "newbing", "stack-claude"]
# P.S. 利用可能な他のモデルには以下も含まれます ["qianfan", "llama2", "qwen", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613",  "gpt-3.5-random"
# "spark", "sparkv2", "chatglm_onnx", "claude-1-100k", "claude-2", "internlm", "jittorllms_pangualpha", "jittorllms_llama"]


# 百度千帆（LLM_MODEL="qianfan"）
BAIDU_CLOUD_API_KEY = ''
BAIDU_CLOUD_SECRET_KEY = ''
BAIDU_CLOUD_QIANFAN_MODEL = 'ERNIE-Bot'    # テキストの翻訳 "ERNIE-Bot"(文心の一言), "ERNIE-Bot-turbo", "BLOOMZ-7B", "Llama-2-70B-Chat", "Llama-2-13B-Chat", "Llama-2-7B-Chat"


# ChatGLM2のファインチューニングモデルを使用するする場合，请把 LLM_MODEL="chatglmft"，テキストの翻訳
CHATGLM_PTUNING_CHECKPOINT = "" # 例如"/home/hmp/ChatGLM2-6B/ptuning/output/6b-pt-128-1e-2/checkpoint-100"


# ローカルのLLMモデル（ChatGLMなど）の実行テキストの翻訳 CPU/GPU
LOCAL_MODEL_DEVICE = "cpu" # テキストの翻訳 "cuda"
LOCAL_MODEL_QUANT = "FP16" # 默认 "FP16" "INT4" 启用量化INT4版本 "INT8" 启用量化INT8版本


# gradioの並行スレッド数を設定する（テキストの翻訳）
CONCURRENT_COUNT = 100


# 送信時に入力ボックスを自動的にクリアしますか
AUTO_CLEAR_TXT = False


# ライブ2Dの装飾を追加する
ADD_WAIFU = False


# 原始文本（テキストの翻訳）（関連機能はしない安定です，gradioのバージョンとネットワークに関連しています，ローカルで使用するする場合はお勧めしません）
# [("username", "password"), ("username2", "password2"), ...]
AUTHENTICATION = []


# 原始文本（通常の場合，テキストの翻訳）（需要配合修改main.py才能生效!）
CUSTOM_PATH = "/"


# HTTPSキーと証明書（テキストの翻訳）
SSL_KEYFILE = ""
SSL_CERTFILE = ""


# 非常にまれなケースでは，openaiの公式KEYは組織コードと一緒に必要です（形式はorg-xxxxxxxxxxxxxxxxxxxxxxxxのようです）使用する
API_ORG = ""


# 原始文本，詳細なチュートリアルはrequest_llm/README.mdを参照してください
SLACK_CLAUDE_BOT_ID = ''   
SLACK_CLAUDE_USER_TOKEN = ''


# テキストの翻訳
AZURE_ENDPOINT = "https://你亲手写的api名称.openai.azure.com/"
AZURE_API_KEY = "Azure OpenAI APIのキーを入力する"    # 原始文本，このオプションはまもなく廃止されます
AZURE_ENGINE = "手書きのデプロイ名を入力してください"            # テキストの翻訳


# Newbingを使用するする (使用するはお勧めしません，将来削除されます)
NEWBING_STYLE = "creative"  # ["creative", "balanced", "precise"]
NEWBING_COOKIES = """
put your new bing cookies here
"""


# 阿里クラウドリアルタイム音声認識の設定は難しく、上級ユーザーのみに使用するをお勧めします 参考 https://github.com/binary-husky/gpt_academic/blob/master/docs/use_audio.md
ENABLE_AUDIO = False
ALIYUN_TOKEN=""     # 例えば、f37f30e0f9934c34a992f6f64f7eba4f
ALIYUN_APPKEY=""    # 例えば RoPlZrM88DnAFkZK
ALIYUN_ACCESSKEY="" # （テキストの翻訳）
ALIYUN_SECRET=""    # （テキストの翻訳）


# 接入讯飞星火大模型 https://console.xfyun.cn/services/iat
XFYUN_APPID = "00000000"
XFYUN_API_SECRET = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
XFYUN_API_KEY = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


# Claude API KEY
ANTHROPIC_API_KEY = ""


# カスタムAPIキーの形式
CUSTOM_API_KEY_PATTERN = ""


# HUGGINGFACE的TOKEN，LLAMAのダウンロード時に有効 https://huggingface.co/docs/hub/security-tokens
HUGGINGFACE_ACCESS_TOKEN = "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"


# GROBIDサーバーアドレス（複数の入力を均等に分散する），テキストの翻訳
# 取得テキストの翻訳：原始文本://huggingface.co/spaces/qingxu98/grobid，publicに設定する，然后GROBID_URL = "https://(原始文本)-(入力したスペース名はgrobidのようです).hf.space"
GROBID_URLS = [
    "https://qingxu98-grobid.hf.space","https://qingxu98-grobid2.hf.space","https://qingxu98-grobid3.hf.space",
    "https://qingxu98-grobid4.hf.space","https://qingxu98-grobid5.hf.space", "https://qingxu98-grobid6.hf.space", 
    "https://qingxu98-grobid7.hf.space", "https://qingxu98-grobid8.hf.space", 
]


# 是否允许通过自然语言描述修改本页テキストの翻訳，该功能具有一定的危险性，デフォルトで閉じる
ALLOW_RESET_CONFIG = False


# 一時的なアップロードフォルダの位置，テキストの翻訳
PATH_PRIVATE_UPLOAD = "private_upload"


# 原始文本，テキストの翻訳
PATH_LOGGING = "gpt_log"


# OpenAIに接続する以外は，テキストの翻訳，テキストの翻訳
WHEN_TO_USE_PROXY = ["Download_LLM", "Download_Gradio_Theme", "Connect_Grobid", "Warmup_Modules"]


# カスタムボタンの最大数制限
NUM_CUSTOM_BASIC_BTN = 4

"""
オンライン大規模モデルの関連関係の構成の示意図
│
├── "gpt-3.5-turbo" 等openai模型
│   ├── API_KEY
│   ├── CUSTOM_API_KEY_PATTERN（テキストの翻訳）
│   ├── API_ORG（テキストの翻訳）
│   └── API_URL_REDIRECT（テキストの翻訳）
│
├── "azure-gpt-3.5" 等azure模型
│   ├── API_KEY
│   ├── AZURE_ENDPOINT
│   ├── AZURE_API_KEY
│   ├── AZURE_ENGINE
│   └── API_URL_REDIRECT
│
├── "spark" 原始文本 spark & sparkv2
│   ├── XFYUN_APPID
│   ├── XFYUN_API_SECRET
│   └── XFYUN_API_KEY
│
├── "claude-1-100k" 等claude模型
│   └── ANTHROPIC_API_KEY
│
├── "stack-claude"
│   ├── SLACK_CLAUDE_BOT_ID
│   └── SLACK_CLAUDE_USER_TOKEN
│
├── "qianfan" 百度千帆大模型库
│   ├── BAIDU_CLOUD_QIANFAN_MODEL
│   ├── BAIDU_CLOUD_API_KEY
│   └── BAIDU_CLOUD_SECRET_KEY
│
├── "newbing" Newbing接口しない再稳定，使用するはお勧めしません
    ├── NEWBING_STYLE
    └── NEWBING_COOKIES

    
ユーザーグラフィカルインターフェースのレイアウト依存関係の示意図
│
├── CHATBOT_HEIGHT チャットボットウィンドウの高さ
├── CODE_HIGHLIGHT コードのハイライト
テキストの翻訳
├── DARK_MODE ダークモード/ライトモード
├── DEFAULT_FN_GROUPS プラグインのデフォルトオプション
テキストの翻訳
テキストの翻訳
追加ワイフにLive2D装飾を追加する
原始文本，该功能具有一定的危险性


プラグインのオンラインサービスの設定依存関係の図
│
├── 音声機能
│   ├── ENABLE_AUDIO
│   ├── ALIYUN_TOKEN
│   ├── ALIYUN_APPKEY
│   ├── ALIYUN_ACCESSKEY
│   └── ALIYUN_SECRET
│
テキストの翻訳
│   └── GROBID_URLS

"""
