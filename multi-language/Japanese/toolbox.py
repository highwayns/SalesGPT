import markdown
import importlib
import time
import inspect
import re
import os
import gradio
import shutil
import glob
from latex2mathml.converter import convert as tex2mathml
from functools import wraps, lru_cache
pj = os.path.join

"""
========================================================================
原始文本
関数プラグインの入出力接続エリア
    - ChatBotWithCookies:   Cookieを持つChatbotクラス，テキストの翻訳
    - ArgsGeneralWrapper:   デコレータ関数，入力パラメータを再構成するために使用するされます，入力パラメータの順序と構造を変更する
    - update_ui:            画面を更新するにはyield from update_uiを使用するします(chatbot, history)
    - CatchException:       プラグインで発生したすべての問題を画面に表示する
    - HotReload:            プラグインのホット更新を実現する
    - trimmed_format_exc:   tracebackを出力する，安全のために絶対アドレスを隠す
========================================================================
"""

class ChatBotWithCookies(list):
    def __init__(self, cookie):
        """
        cookies = {
            'top_p': top_p,
            'temperature': temperature,
            'lock_plugin': bool,
            "files_to_promote": ["file1", "file2"],
            "most_recent_uploaded": {
                "path": "uploaded_path",
                "time": time.time(),
                "time_str": "timestr",
            }
        }
        """
        self._cookies = cookie

    def write_list(self, list):
        for t in list:
            self.append(t)

    def get_list(self):
        return [t for t in self]

    def get_cookies(self):
        return self._cookies


def ArgsGeneralWrapper(f):
    """
    デコレータ関数，入力パラメータを再構成するために使用するされます，入力パラメータの順序と構造を変更する。
    """
    def decorated(request: gradio.Request, cookies, max_length, llm_model, txt, txt2, top_p, temperature, chatbot, history, system_prompt, plugin_advanced_arg, *args):
        txt_passon = txt
        if txt == "" and txt2 != "": txt_passon = txt2
        # 引入一テキストの翻訳有cookie的chatbot
        cookies.update({
            'top_p':top_p,
            'api_key': cookies['api_key'],
            'llm_model': llm_model,
            'temperature':temperature,
        })
        llm_kwargs = {
            'api_key': cookies['api_key'],
            'llm_model': llm_model,
            'top_p':top_p,
            'max_length': max_length,
            'temperature':temperature,
            'client_ip': request.client.host,
        }
        plugin_kwargs = {
            "advanced_arg": plugin_advanced_arg,
        }
        chatbot_with_cookie = ChatBotWithCookies(cookies)
        chatbot_with_cookie.write_list(chatbot)
        
        if cookies.get('lock_plugin', None) is None:
            # 正常な状態
            if len(args) == 0:  # プラグインチャネル
                yield from f(txt_passon, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, system_prompt, request)
            else:               # テキストの翻訳，または基本機能チャネル
                yield from f(txt_passon, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, system_prompt, *args)
        else:
            # テキストの翻訳
            module, fn_name = cookies['lock_plugin'].split('->')
            f_hot_reload = getattr(importlib.import_module(module, fn_name), fn_name)
            yield from f_hot_reload(txt_passon, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, system_prompt, request)
            # ユーザーが誤って対話チャネルを介して入力したかどうかを判断する，もし，それに注意してください
            final_cookies = chatbot_with_cookie.get_cookies()
            # len(args) != 0は「送信」キーの対話チャネルを表します，または基本機能チャネル
            if len(args) != 0 and 'files_to_promote' in final_cookies and len(final_cookies['files_to_promote']) > 0:
                chatbot_with_cookie.append(["**滞留しているキャッシュドキュメント**が検出されました，迅速に処理してください。", "すべての保留ドキュメントを取得するには、**現在の対話を保存**をクリックしてください。"])
                yield from update_ui(chatbot_with_cookie, final_cookies['history'], msg="キャッシュドキュメントの滞留が検出されました")
    return decorated


def update_ui(chatbot, history, msg='正常', **kwargs):  # 画面をリフレッシュする
    """
    ユーザーインターフェースをリフレッシュする
    """
    assert isinstance(chatbot, ChatBotWithCookies), "テキストの翻訳。テキストの翻訳, clearを使用するしてそれをクリアできます, その後、for+appendループを使用するして再割り当てする。"
    cookies = chatbot.get_cookies()
    # テキストの翻訳
    cookies.update({'history': history})
    # プラグインのロック時のインターフェース表示の問題を解決する
    if cookies.get('lock_plugin', None):
        label = cookies.get('llm_model', "") + " | " + "プラグインをロックしています" + cookies.get('lock_plugin', None)
        chatbot_gr = gradio.update(value=chatbot, label=label)
        if cookies.get('label', "") != label: cookies['label'] = label   # 记住当前的label
    elif cookies.get('label', None):
        chatbot_gr = gradio.update(value=chatbot, label=cookies.get('llm_model', ""))
        cookies['label'] = None    # テキストの翻訳
    else:
        chatbot_gr = chatbot

    yield cookies, chatbot_gr, history, msg

def update_ui_lastest_msg(lastmsg, chatbot, history, delay=1):  # 画面をリフレッシュする
    """
    ユーザーインターフェースをリフレッシュする
    """
    if len(chatbot) == 0: chatbot.append(["update_ui_last_msg", lastmsg])
    chatbot[-1] = list(chatbot[-1])
    chatbot[-1][-1] = lastmsg
    yield from update_ui(chatbot=chatbot, history=history)
    time.sleep(delay)


def trimmed_format_exc():
    import os, traceback
    str = traceback.format_exc()
    current_path = os.getcwd()
    replace_path = "."
    return str.replace(current_path, replace_path)

def CatchException(f):
    """
    デコレータ関数，関数fでの例外をキャッチし、ジェネレータにパッケージ化して返す，そしてチャットに表示されます。
    """

    @wraps(f)
    def decorated(main_input, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, *args, **kwargs):
        try:
            yield from f(main_input, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, *args, **kwargs)
        except Exception as e:
            from check_proxy import check_proxy
            from toolbox import get_conf
            proxies, = get_conf('proxies')
            tb_str = '```\n' + trimmed_format_exc() + '```'
            if len(chatbot_with_cookie) == 0:
                chatbot_with_cookie.clear()
                chatbot_with_cookie.append(["プラグインのスケジュール例外", "原始文本"])
            chatbot_with_cookie[-1] = (chatbot_with_cookie[-1][0],
                           f"[Local Message] 原始文本: \n\n{tb_str} \n\n現在のプロキシの可用性: \n\n{check_proxy(proxies)}")
            yield from update_ui(chatbot=chatbot_with_cookie, history=history, msg=f'例外 {e}') # 画面をリフレッシュする
    return decorated


def HotReload(f):
    """
    HotReloadのデコレータ関数，Python関数プラグインのホットアップデートに使用するされます。
    関数のホットアップデートは、プログラムの実行を停止せずに行われることを指します，原始文本，これにより、リアルタイムの更新機能が実現されます。
    テキストの翻訳，wrapsを使用するする(f)関数のメタ情報を保持するために，decoratedという名前の内部関数を定義する。
    テキストの翻訳，
    そして、getattr関数を使用するして関数名を取得します，テキストの翻訳。
    原始文本，テキストの翻訳，テキストの翻訳。
    原始文本，デコレータ関数は内部関数を返す。この内部関数は、関数の元の定義を最新バージョンに更新することができます，新しいバージョンの関数を実行する。
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        fn_name = f.__name__
        f_hot_reload = getattr(importlib.reload(inspect.getmodule(f)), fn_name)
        yield from f_hot_reload(*args, **kwargs)
    return decorated


"""
========================================================================
第2部分
その他のツール:
    - write_history_to_file:    テキストの翻訳
    - regular_txt_to_markdown:  通常のテキストをMarkdown形式のテキストに変換します。
    - report_execption:         チャットボットに簡単な予期しないエラーメッセージを追加する
    - text_divide_paragraph:    将文本按照段落分隔符分割开，段落タグを含むHTMLコードを生成する。
    - markdown_convertion:      テキストの翻訳，Markdownを美しいHTMLに変換する
    - format_io:                gradioのデフォルトのMarkdown処理テキストの翻訳を上書きする
    - on_file_uploaded:         ファイルのアップロードを処理する（自動解凍）
    - on_report_generated:      生成されたレポートを自動的にファイルアップロードエリアに投影します
    - clip_history:             過去のコンテキストが長すぎる場合，自動的に切り捨てる
    - get_conf:                 設定の取得
    - select_api_key:           テキストの翻訳，利用可能なAPIキーを抽出する
========================================================================
"""

def get_reduce_token_percent(text):
    """
        * この関数は将来的に廃止されます
    """
    try:
        # text = "maximum context length is 4097 tokens. However, your messages resulted in 4870 tokens"
        pattern = r"(\d+)\s+tokens\b"
        match = re.findall(pattern, text)
        EXCEED_ALLO = 500  # 少し余地を残してください，それ以外の場合、返信時に問題が発生する可能性があります
        max_limit = float(match[0]) - EXCEED_ALLO
        current_tokens = float(match[1])
        ratio = max_limit/current_tokens
        assert ratio > 0 and ratio < 1
        return ratio, str(int(current_tokens-max_limit))
    except:
        return 0.5, '詳細しない明'


def write_history_to_file(history, file_basename=None, file_fullname=None, auto_caption=True):
    """
    対話履歴をMarkdown形式でファイルに書き込む。テキストの翻訳，現在の時間を使用するしてファイル名を生成する。
    """
    import os
    import time
    if file_fullname is None:
        if file_basename is not None:
            file_fullname = pj(get_log_folder(), file_basename)
        else:
            file_fullname = pj(get_log_folder(), f'GPT-Academic-{gen_time_str()}.md')
    os.makedirs(os.path.dirname(file_fullname), exist_ok=True)
    with open(file_fullname, 'w', encoding='utf8') as f:
        f.write('# GPT-Academic Report\n')
        for i, content in enumerate(history):
            try:    
                if type(content) != str: content = str(content)
            except:
                continue
            if i % 2 == 0 and auto_caption:
                f.write('## ')
            try:
                f.write(content)
            except:
                # remove everything that cannot be handled by utf8
                f.write(content.encode('utf-8', 'ignore').decode())
            f.write('\n\n')
    res = os.path.abspath(file_fullname)
    return res


def regular_txt_to_markdown(text):
    """
    通常のテキストをMarkdown形式のテキストに変換します。
    """
    text = text.replace('\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text




def report_execption(chatbot, history, a, b):
    """
    chatbotにエラーメッセージを追加する
    """
    chatbot.append((a, b))
    history.extend([a, b])


def text_divide_paragraph(text):
    """
    将文本按照段落分隔符分割开，段落タグを含むHTMLコードを生成する。
    """
    pre = '<div class="markdown-body">'
    suf = '</div>'
    if text.startswith(pre) and text.endswith(suf):
        return text
    
    if '```' in text:
        # careful input
        return pre + text + suf
    else:
        # wtf input
        lines = text.split("\n")
        for i, line in enumerate(lines):
            lines[i] = lines[i].replace(" ", "&nbsp;")
        text = "</br>".join(lines)
        return pre + text + suf


@lru_cache(maxsize=128) # lruキャッシュを使用するして変換速度を高速化する
def markdown_convertion(txt):
    """
    Markdown形式のテキストをHTML形式に変換する。数学式が含まれている場合，テキストの翻訳。
    """
    pre = '<div class="markdown-body">'
    suf = '</div>'
    if txt.startswith(pre) and txt.endswith(suf):
        # print('警告，変換済みの文字列を入力しました，二次转化可能出問題')
        return txt # 既に変換されています，再変換はしない要です
    
    markdown_extension_configs = {
        'mdx_math': {
            'enable_dollar_delimiter': True,
            'use_gitlab_delimiters': False,
        },
    }
    find_equation_pattern = r'<script type="math/tex(?:.*?)>(.*?)</script>'

    def tex2mathml_catch_exception(content, *args, **kwargs):
        try:
            content = tex2mathml(content, *args, **kwargs)
        except:
            content = content
        return content

    def replace_math_no_render(match):
        content = match.group(1)
        if 'mode=display' in match.group(0):
            content = content.replace('\n', '</br>')
            return f"<font color=\"#00FF00\">$$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$$</font>"
        else:
            return f"<font color=\"#00FF00\">$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$</font>"

    def replace_math_render(match):
        content = match.group(1)
        if 'mode=display' in match.group(0):
            if '\\begin{aligned}' in content:
                content = content.replace('\\begin{aligned}', '\\begin{array}')
                content = content.replace('\\end{aligned}', '\\end{array}')
                content = content.replace('&', ' ')
            content = tex2mathml_catch_exception(content, display="block")
            return content
        else:
            return tex2mathml_catch_exception(content)

    def markdown_bug_hunt(content):
        """
        mdx_mathのバグを解決する（beginコマンドを単一の$で囲む必要はありません<script>）
        """
        content = content.replace('<script type="math/tex">\n<script type="math/tex; mode=display">', '<script type="math/tex; mode=display">')
        content = content.replace('</script>\n</script>', '</script>')
        return content

    def is_equation(txt):
        """
        公式であるかどうかを判定する | テスト1 ローレンツ力を書く，TeX形式の数式を使用するしてテスト2でコーシーのしない等式を示す，LaTeX形式でテスト3を使用するしてマクスウェル方程式を書く
        """
        if '```' in txt and '```reference' not in txt: return False
        if '$' not in txt and '\\[' not in txt: return False
        mathpatterns = {
            r'(?<!\\|\$)(\$)([^\$]+)(\$)': {'allow_multi_lines': False},                            #  $...$
            r'(?<!\\)(\$\$)([^\$]+)(\$\$)': {'allow_multi_lines': True},                            # $$...$$
            r'(?<!\\)(\\\[)(.+?)(\\\])': {'allow_multi_lines': False},                              # \[...\]
            # r'(?<!\\)(\\\()(.+?)(\\\))': {'allow_multi_lines': False},                            # \(...\)
            # r'(?<!\\)(\\begin{([a-z]+?\*?)})(.+?)(\\end{\2})': {'allow_multi_lines': True},       # \begin...\end
            # r'(?<!\\)(\$`)([^`]+)(`\$)': {'allow_multi_lines': False},                            # $`...`$
        }
        matches = []
        for pattern, property in mathpatterns.items():
            flags = re.ASCII|re.DOTALL if property['allow_multi_lines'] else re.ASCII
            matches.extend(re.findall(pattern, txt, flags))
        if len(matches) == 0: return False
        contain_any_eq = False
        illegal_pattern = re.compile(r'[^\x00-\x7F]|echo')
        for match in matches:
            if len(match) != 3: return False
            eq_canidate = match[1]
            if illegal_pattern.search(eq_canidate): 
                return False
            else: 
                contain_any_eq = True
        return contain_any_eq

    if is_equation(txt):  # $でマークされた数式記号があります，コードセグメントはありません```テキストの翻訳
        # convert everything to html format
        split = markdown.markdown(text='---')
        convert_stage_1 = markdown.markdown(text=txt, extensions=['sane_lists', 'tables', 'mdx_math', 'fenced_code'], extension_configs=markdown_extension_configs)
        convert_stage_1 = markdown_bug_hunt(convert_stage_1)
        # 1. convert to easy-to-copy tex (do not render math)
        convert_stage_2_1, n = re.subn(find_equation_pattern, replace_math_no_render, convert_stage_1, flags=re.DOTALL)
        # 2. convert to rendered equation
        convert_stage_2_2, n = re.subn(find_equation_pattern, replace_math_render, convert_stage_1, flags=re.DOTALL)
        # cat them together
        return pre + convert_stage_2_1 + f'{split}' + convert_stage_2_2 + suf
    else:
        return pre + markdown.markdown(txt, extensions=['sane_lists', 'tables', 'fenced_code', 'codehilite']) + suf


def close_up_code_segment_during_stream(gpt_reply):
    """
    GPTの出力コードの途中で（前のものを出力しました```，ただし、まだ後半部分を出力していません```），後ろを補完します```

    Args:
        gpt_reply (str): GPTモデルからの応テキストの翻訳文字列。

    Returns:
        str: 新しい文字列を返します，テキストの翻訳```補完。

    """
    if '```' not in gpt_reply:
        return gpt_reply
    if gpt_reply.endswith('```'):
        return gpt_reply

    # 排除了以上两テキストの翻訳情况，私たち
    segments = gpt_reply.split('```')
    n_mark = len(segments) - 1
    if n_mark % 2 == 1:
        # print('输出代码片段中！')
        return gpt_reply+'\n```'
    else:
        return gpt_reply


def format_io(self, y):
    """
    入力と出力をHTML形式で解析します。原始文本，そして、出力の一部をMarkdownと数式をHTML形式に変換します。
    """
    if y is None or y == []:
        return []
    i_ask, gpt_reply = y[-1]
    # 入力部分が自由すぎます，前処理を行う
    if i_ask is not None: i_ask = text_divide_paragraph(i_ask)
    # コードの出力が途中で切れる場合，後のものを補うことを試してみてください```
    if gpt_reply is not None: gpt_reply = close_up_code_segment_during_stream(gpt_reply)
    # process
    y[-1] = (
        None if i_ask is None else markdown.markdown(i_ask, extensions=['fenced_code', 'tables']),
        None if gpt_reply is None else markdown_convertion(gpt_reply)
    )
    return y


def find_free_port():
    """
    返回当前系统中可用的未使用するポート。
    """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def extract_archive(file_path, dest_dir):
    import zipfile
    import tarfile
    import os
    # Get the file extension of the input file
    file_extension = os.path.splitext(file_path)[1]

    # Extract the archive based on its extension
    if file_extension == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zipobj:
            zipobj.extractall(path=dest_dir)
            print("Successfully extracted zip archive to {}".format(dest_dir))

    elif file_extension in ['.tar', '.gz', '.bz2']:
        with tarfile.open(file_path, 'r:*') as tarobj:
            tarobj.extractall(path=dest_dir)
            print("Successfully extracted tar archive to {}".format(dest_dir))

    # 第三方库，事前にpip install rarfileが必要です
    # テキストの翻訳，Windows上ではwinrarソフトウェアもインストールする必要があります，そのPath環境変数を設定する，如"C:\Program Files\WinRAR"才可以
    elif file_extension == '.rar':
        try:
            import rarfile
            with rarfile.RarFile(file_path) as rf:
                rf.extractall(path=dest_dir)
                print("Successfully extracted rar archive to {}".format(dest_dir))
        except:
            print("Rar format requires additional dependencies to install")
            return '\n\n解凍に失敗しました！ rarファイルを解凍するにはpip install rarfileをインストールする必要があります。アドバイス：zip圧縮形式を使用するします。'

    # 第三方库，事前にpip install py7zrが必要です
    elif file_extension == '.7z':
        try:
            import py7zr
            with py7zr.SevenZipFile(file_path, mode='r') as f:
                f.extractall(path=dest_dir)
                print("Successfully extracted 7z archive to {}".format(dest_dir))
        except:
            print("7z format requires additional dependencies to install")
            return '\n\n解凍に失敗しました！ 7zファイルを解凍するには、pip install py7zrをインストールする必要があります'
    else:
        return ''
    return ''


def find_recent_files(directory):
    """
        me: find files that is created with in one minutes under a directory with python, write a function
        gpt: here it is!
    """
    import os
    import time
    current_time = time.time()
    one_minute_ago = current_time - 60
    recent_files = []
    if not os.path.exists(directory): 
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = pj(directory, filename)
        if file_path.endswith('.log'):
            continue
        created_time = os.path.getmtime(file_path)
        if created_time >= one_minute_ago:
            if os.path.isdir(file_path):
                continue
            recent_files.append(file_path)

    return recent_files

def promote_file_to_downloadzone(file, rename_file=None, chatbot=None):
    # ファイルをダウンロードエリアにコピーする
    import shutil
    if rename_file is None: rename_file = f'{gen_time_str()}-{os.path.basename(file)}'
    new_path = pj(get_log_folder(), rename_file)
    # 既に存在する場合，まず削除してください
    if os.path.exists(new_path) and not os.path.samefile(new_path, file): os.remove(new_path)
    # ファイルをコピーする
    if not os.path.exists(new_path): shutil.copyfile(file, new_path)
    # ファイルをchatbotのクッキーに追加する，複数のユーザーによる干渉を避ける
    if chatbot is not None:
        if 'files_to_promote' in chatbot._cookies: current = chatbot._cookies['files_to_promote']
        else: current = []
        chatbot._cookies.update({'files_to_promote': [new_path] + current})
    return new_path

def disable_auto_promotion(chatbot):
    chatbot._cookies.update({'files_to_promote': []})
    return

def is_the_upload_folder(string):
    PATH_PRIVATE_UPLOAD, = get_conf('PATH_PRIVATE_UPLOAD')
    pattern = r'^PATH_PRIVATE_UPLOAD/[A-Za-z0-9_-]+/\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'
    pattern = pattern.replace('PATH_PRIVATE_UPLOAD', PATH_PRIVATE_UPLOAD)
    if re.match(pattern, string): return True
    else: return False

def del_outdated_uploads(outdate_time_seconds):
    PATH_PRIVATE_UPLOAD, = get_conf('PATH_PRIVATE_UPLOAD')
    current_time = time.time()
    one_hour_ago = current_time - outdate_time_seconds
    # Get a list of all subdirectories in the PATH_PRIVATE_UPLOAD folder
    # Remove subdirectories that are older than one hour
    for subdirectory in glob.glob(f'{PATH_PRIVATE_UPLOAD}/*/*'):
        subdirectory_time = os.path.getmtime(subdirectory)
        if subdirectory_time < one_hour_ago:
            try: shutil.rmtree(subdirectory)
            except: pass
    return

def on_file_uploaded(request: gradio.Request, files, chatbot, txt, txt2, checkboxes, cookies):
    """
    ファイルがアップロードされたときのコールバック関数
    """
    if len(files) == 0:
        return chatbot, txt
    
    # 原始文本
    outdate_time_seconds = 60
    del_outdated_uploads(outdate_time_seconds)

    # 原始文本
    user_name = "default" if not request.username else request.username
    time_tag = gen_time_str()
    PATH_PRIVATE_UPLOAD, = get_conf('PATH_PRIVATE_UPLOAD')
    target_path_base = pj(PATH_PRIVATE_UPLOAD, user_name, time_tag)
    os.makedirs(target_path_base, exist_ok=True)

    # ファイルを1つずつ目標パスに移動する
    upload_msg = ''
    for file in files:
        file_origin_name = os.path.basename(file.orig_name)
        this_file_path = pj(target_path_base, file_origin_name)
        shutil.move(file.name, this_file_path)
        upload_msg += extract_archive(file_path=this_file_path, dest_dir=this_file_path+'.extract')
    
    # ファイルの整理
    moved_files = [fp for fp in glob.glob(f'{target_path_base}/**/*', recursive=True)]
    if "フローティング入力エリア" in checkboxes: 
        txt, txt2 = "", target_path_base
    else:
        txt, txt2 = target_path_base, ""

    # 原始文本
    moved_files_str = '\t\n\n'.join(moved_files)
    chatbot.append(['テキストの翻訳，ご確認ください', 
                    f'[Local Message] テキストの翻訳: \n\n{moved_files_str}' +
                    f'\n\n呼び出しパスのパラメータが自動的に修正されました: \n\n{txt}' +
                    f'\n\n今、任意の関数プラグインをクリックすると，上記のファイルは入力パラメータとして使用するされます'+upload_msg])
    
    # 原始文本
    cookies.update({
        'most_recent_uploaded': {
            'path': target_path_base,
            'time': time.time(),
            'time_str': time_tag
    }})
    return chatbot, txt, txt2, cookies


def on_report_generated(cookies, files, chatbot):
    from toolbox import find_recent_files
    PATH_LOGGING, = get_conf('PATH_LOGGING')
    if 'files_to_promote' in cookies:
        report_files = cookies['files_to_promote']
        cookies.pop('files_to_promote')
    else:
        report_files = find_recent_files(PATH_LOGGING)
    if len(report_files) == 0:
        return cookies, None, chatbot
    # files.extend(report_files)
    file_links = ''
    for f in report_files: file_links += f'<br/><a href="file={os.path.abspath(f)}" target="_blank">{f}</a>'
    chatbot.append(['リモートでの取得テキストの翻訳を報告する？', f'テキストの翻訳（折りたたまれた状態になっている可能性があります），ご確認ください。{file_links}'])
    return cookies, report_files, chatbot

def load_chat_cookies():
    API_KEY, LLM_MODEL, AZURE_API_KEY = get_conf('API_KEY', 'LLM_MODEL', 'AZURE_API_KEY')
    DARK_MODE, NUM_CUSTOM_BASIC_BTN = get_conf('DARK_MODE', 'NUM_CUSTOM_BASIC_BTN')
    if is_any_api_key(AZURE_API_KEY):
        if is_any_api_key(API_KEY): API_KEY = API_KEY + ',' + AZURE_API_KEY
        else: API_KEY = AZURE_API_KEY
    customize_fn_overwrite_ = {}
    for k in range(NUM_CUSTOM_BASIC_BTN):
        customize_fn_overwrite_.update({  
            "原始文本" + str(k+1):{
                "Title":    r"",
                "Prefix":   r"カスタムメニューでヒントの接頭辞を定義してください.",
                "Suffix":   r"カスタムメニューでヒントの接尾辞を定義してください",
            }
        })
    return {'api_key': API_KEY, 'llm_model': LLM_MODEL, 'customize_fn_overwrite': customize_fn_overwrite_}

def is_openai_api_key(key):
    CUSTOM_API_KEY_PATTERN, = get_conf('CUSTOM_API_KEY_PATTERN')
    if len(CUSTOM_API_KEY_PATTERN) != 0:
        API_MATCH_ORIGINAL = re.match(CUSTOM_API_KEY_PATTERN, key)
    else:
        API_MATCH_ORIGINAL = re.match(r"sk-[a-zA-Z0-9]{48}$", key)
    return bool(API_MATCH_ORIGINAL)

def is_azure_api_key(key):
    API_MATCH_AZURE = re.match(r"[a-zA-Z0-9]{32}$", key)
    return bool(API_MATCH_AZURE)

def is_api2d_key(key):
    API_MATCH_API2D = re.match(r"fk[a-zA-Z0-9]{6}-[a-zA-Z0-9]{32}$", key)
    return bool(API_MATCH_API2D)

def is_any_api_key(key):
    if ',' in key:
        keys = key.split(',')
        for k in keys:
            if is_any_api_key(k): return True
        return False
    else:
        return is_openai_api_key(key) or is_api2d_key(key) or is_azure_api_key(key)

def what_keys(keys):
    avail_key_list = {'OpenAI Key':0, "Azure Key":0, "API2D Key":0}
    key_list = keys.split(',')

    for k in key_list:
        if is_openai_api_key(k): 
            avail_key_list['OpenAI Key'] += 1

    for k in key_list:
        if is_api2d_key(k): 
            avail_key_list['API2D Key'] += 1

    for k in key_list:
        if is_azure_api_key(k): 
            avail_key_list['Azure Key'] += 1

    return f"検出されました： OpenAI Key {avail_key_list['OpenAI Key']} テキストの翻訳, Azure Key {avail_key_list['Azure Key']} テキストの翻訳, API2D Key {avail_key_list['API2D Key']} テキストの翻訳"

def select_api_key(keys, llm_model):
    import random
    avail_key_list = []
    key_list = keys.split(',')

    if llm_model.startswith('gpt-'):
        for k in key_list:
            if is_openai_api_key(k): avail_key_list.append(k)

    if llm_model.startswith('api2d-'):
        for k in key_list:
            if is_api2d_key(k): avail_key_list.append(k)

    if llm_model.startswith('azure-'):
        for k in key_list:
            if is_azure_api_key(k): avail_key_list.append(k)

    if len(avail_key_list) == 0:
        raise RuntimeError(f"提供されたAPIキーは要件を満たしていません，使用するできるものは含まれていません{llm_model}のAPIキー。原始文本（右下隅のモデルメニューでopenaiを切り替えることができます,azure,claude,api2dなどのリクエストソース）。")

    api_key = random.choice(avail_key_list) # テキストの翻訳
    return api_key

def read_env_variable(arg, default_value):
    """
    環境変数は次のようになります `GPT_ACADEMIC_CONFIG`(優先)，也可以直接是`CONFIG`
    たとえば、Windowsのコマンドプロンプトで，テキストの翻訳：
        set USE_PROXY=True
        set API_KEY=sk-j7caBpkRoxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        set proxies={"http":"http://127.0.0.1:10085", "https":"http://127.0.0.1:10085",}
        set AVAIL_LLM_MODELS=["gpt-3.5-turbo", "chatglm"]
        set AUTHENTICATION=[("username", "password"), ("username2", "password2")]
    原始文本：
        set GPT_ACADEMIC_USE_PROXY=True
        set GPT_ACADEMIC_API_KEY=sk-j7caBpkRoxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        set GPT_ACADEMIC_proxies={"http":"http://127.0.0.1:10085", "https":"http://127.0.0.1:10085",}
        set GPT_ACADEMIC_AVAIL_LLM_MODELS=["gpt-3.5-turbo", "chatglm"]
        set GPT_ACADEMIC_AUTHENTICATION=[("username", "password"), ("username2", "password2")]
    """
    from colorful import printBrightRed, printBrightGreen
    arg_with_prefix = "GPT_ACADEMIC_" + arg 
    if arg_with_prefix in os.environ: 
        env_arg = os.environ[arg_with_prefix]
    elif arg in os.environ: 
        env_arg = os.environ[arg]
    else:
        raise KeyError
    print(f"[ENV_VAR] 読み込みを試みています{arg}，原始文本：{default_value} --> 修正値：{env_arg}")
    try:
        if isinstance(default_value, bool):
            env_arg = env_arg.strip()
            if env_arg == 'True': r = True
            elif env_arg == 'False': r = False
            else: print('enter True or False, but have:', env_arg); r = default_value
        elif isinstance(default_value, int):
            r = int(env_arg)
        elif isinstance(default_value, float):
            r = float(env_arg)
        elif isinstance(default_value, str):
            r = env_arg.strip()
        elif isinstance(default_value, dict):
            r = eval(env_arg)
        elif isinstance(default_value, list):
            r = eval(env_arg)
        elif default_value is None:
            assert arg == "proxies"
            r = eval(env_arg)
        else:
            printBrightRed(f"[ENV_VAR] テキストの翻訳{arg}環境変数を介した設定はサポートされていません！ ")
            raise KeyError
    except:
        printBrightRed(f"[ENV_VAR] テキストの翻訳{arg}読み込みに失敗しました！ ")
        raise KeyError(f"[ENV_VAR] テキストの翻訳{arg}読み込みに失敗しました！ ")

    printBrightGreen(f"[ENV_VAR] 成功読み取り環境変数{arg}")
    return r

@lru_cache(maxsize=128)
def read_single_conf_with_lru_cache(arg):
    from colorful import printBrightRed, printBrightGreen, printBrightBlue
    try:
        # 優先度1. 環境変数を設定として取得する
        default_ref = getattr(importlib.import_module('config'), arg)   # テキストの翻訳
        r = read_env_variable(arg, default_ref) 
    except:
        try:
            # 優先度2. config_privateから設定を取得する
            r = getattr(importlib.import_module('config_private'), arg)
        except:
            # 優先度3. configから設定を取得する
            r = getattr(importlib.import_module('config'), arg)

    # API_KEYの読み取り時に，configを変更するのを忘れていないか確認してください
    if arg == 'API_KEY':
        printBrightBlue(f"[API_KEY] このプロジェクトは、OpenAIおよびAzureのAPIキーをサポートしています。也支持同時填写多テキストの翻訳api-key，如API_KEY=\"openai-key1,openai-key2,azure-key3\"")
        printBrightBlue(f"[API_KEY] config.pyでapi-keyを変更することもできます(s)，また、質問入力エリアに一時的なAPIキーを入力することもできます(s)，Enterキーを押して送信すると有効になります。")
        if is_any_api_key(r):
            printBrightGreen(f"[API_KEY] API_KEYは: {r[:15]}原始文本")
        else:
            printBrightRed( "[API_KEY] API_KEYが既知のいずれのキーフォーマットにも満たしていません，configファイルでAPIキーを変更してから再実行してください。")
    if arg == 'proxies':
        if not read_single_conf_with_lru_cache('USE_PROXY'): r = None   # USE_PROXYをチェックする，原始文本
        if r is None:
            printBrightRed('[PROXY] ネットワークプロキシの状態：設定されていない。プロキシなしの状態では、OpenAIファミリのモデルにアクセスできない可能性があります。アドバイス：USE_PROXYをチェックする选项是否修改。')
        else:
            printBrightGreen('[PROXY] ネットワークプロキシの状態：設定済み。以下の設定情報：', r)
            assert isinstance(r, dict), 'プロキシの形式が間違っています，proxiesオプションの形式に注意してください，括弧を忘れないでください。'
    return r


@lru_cache(maxsize=128)
def get_conf(*args):
    # 原始文本, 如APIと代理网址, 誤ってgithubに渡されて他の人に見られるのを避ける
    res = []
    for arg in args:
        r = read_single_conf_with_lru_cache(arg)
        res.append(r)
    return res


def clear_line_break(txt):
    txt = txt.replace('\n', ' ')
    txt = txt.replace('  ', ' ')
    txt = txt.replace('  ', ' ')
    return txt


class DummyWith():
    """
    このコードは、DummyWithという名前の空のコンテキストマネージャを定義しています，
    テキストの翻訳，コード構造が変わらないように他のコンテキストマネージャを置き換える。
    コンテキストマネージャはPythonオブジェクトです，with文と一緒に使用するするための，
    原始文本。
    コンテキストマネージャは2つのメソッドを実装する必要があります，__enter__それぞれ()と __exit__()。
    テキストの翻訳，__enter__()原始文本，
    そして、コンテキストの実行が終了した時に，__exit__()メソッドが呼び出されます。
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

def run_gradio_in_subpath(demo, auth, port, custom_path):
    """
    gradioの実行アドレスを指定のセカンダリパスに変更する
    """
    def is_path_legal(path: str)->bool:
        '''
        check path for sub url
        path: path to check
        return value: do sub url wrap
        '''
        if path == "/": return True
        if len(path) == 0:
            print("ilegal custom path: {}\npath must not be empty\ndeploy on root url".format(path))
            return False
        if path[0] == '/':
            if path[1] != '/':
                print("deploy on sub-path {}".format(path))
                return True
            return False
        print("ilegal custom path: {}\npath should begin with \'/\'\ndeploy on root url".format(path))
        return False

    if not is_path_legal(custom_path): raise RuntimeError('Ilegal custom path')
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI
    app = FastAPI()
    if custom_path != "/":
        @app.get("/")
        def read_main(): 
            return {"message": f"Gradio is running at: {custom_path}"}
    app = gr.mount_gradio_app(app, demo, path=custom_path)
    uvicorn.run(app, host="0.0.0.0", port=port) # , auth=auth


def clip_history(inputs, history, tokenizer, max_token_limit):
    """
    reduce the length of history by clipping.
    this function search for the longest entries to clip, little by little,
    until the number of token of history is reduced under threshold.
    履歴の長さを短くするためにトリミングする。 
    この関数は最長のエントリを順番に検索してトリミングします，
    直到テキストの翻訳记录的标记数量降低到阈值以下。
    """
    import numpy as np
    from request_llm.bridge_all import model_info
    def get_token_num(txt): 
        return len(tokenizer.encode(txt, disallowed_special=()))
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit * 3 / 4:
        # 入力部分のトークンの割合が制限の3/4未満の場合，トリミング時
        # 1. 入力の余裕を残す
        max_token_limit = max_token_limit - input_token_num
        # テキストの翻訳
        max_token_limit = max_token_limit - 128
        # 3. もし余剰が小さすぎる場合，履歴を直接クリアする
        if max_token_limit < 128:
            history = []
            return history
    else:
        # 入力部分のトークンの割合が > 制限の3/4の時，履歴を直接クリアする
        history = []
        return history

    everything = ['']
    everything.extend(history)
    n_token = get_token_num('\n'.join(everything))
    everything_token = [get_token_num(e) for e in everything]

    # 切り捨て時の粒度
    delta = max(everything_token) // 16

    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = tokenizer.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]
        everything[where] = tokenizer.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))

    history = everything[1:]
    return history

"""
========================================================================
テキストの翻訳
その他のツール:
    - zip_folder:    指定したパスのすべてのファイルを圧縮する，テキストの翻訳（gptで書かれた）
    - gen_time_str:  テキストの翻訳
    - ProxyNetworkActivate: テキストの翻訳（如果有）
    - objdump/objload: 便利なデバッグ関数
========================================================================
"""

def zip_folder(source_folder, dest_folder, zip_name):
    import zipfile
    import os
    # Make sure the source folder exists
    if not os.path.exists(source_folder):
        print(f"{source_folder} does not exist")
        return

    # Make sure the destination folder exists
    if not os.path.exists(dest_folder):
        print(f"{dest_folder} does not exist")
        return

    # Create the name for the zip file
    zip_file = pj(dest_folder, zip_name)

    # Create a ZipFile object
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the source folder and add files to the zip file
        for foldername, subfolders, filenames in os.walk(source_folder):
            for filename in filenames:
                filepath = pj(foldername, filename)
                zipf.write(filepath, arcname=os.path.relpath(filepath, source_folder))

    # Move the zip file to the destination folder (if it wasn't already there)
    if os.path.dirname(zip_file) != dest_folder:
        os.rename(zip_file, pj(dest_folder, os.path.basename(zip_file)))
        zip_file = pj(dest_folder, os.path.basename(zip_file))

    print(f"Zip file created at {zip_file}")

def zip_result(folder):
    t = gen_time_str()
    zip_folder(folder, get_log_folder(), f'{t}-result.zip')
    return pj(get_log_folder(), f'{t}-result.zip')

def gen_time_str():
    import time
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_log_folder(user='default', plugin_name='shared'):
    PATH_LOGGING, = get_conf('PATH_LOGGING')
    _dir = pj(PATH_LOGGING, user, plugin_name)
    if not os.path.exists(_dir): os.makedirs(_dir)
    return _dir

class ProxyNetworkActivate():
    """
    このコードはTempProxyという名前の空のコンテキストマネージャを定義しています, 一部のコードにプロキシを設定するために使用するされます
    """
    def __init__(self, task=None) -> None:
        self.task = task
        if not task:
            # タスクが指定されていない, 那么私たち默认代理生效
            self.valid = True
        else:
            # タスクが指定されました, 原始文本
            from toolbox import get_conf
            WHEN_TO_USE_PROXY, = get_conf('WHEN_TO_USE_PROXY')
            self.valid = (task in WHEN_TO_USE_PROXY)

    def __enter__(self):
        if not self.valid: return self
        from toolbox import get_conf
        proxies, = get_conf('proxies')
        if 'no_proxy' in os.environ: os.environ.pop('no_proxy')
        if proxies is not None:
            if 'http' in proxies: os.environ['HTTP_PROXY'] = proxies['http']
            if 'https' in proxies: os.environ['HTTPS_PROXY'] = proxies['https']
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ['no_proxy'] = '*'
        if 'HTTP_PROXY' in os.environ: os.environ.pop('HTTP_PROXY')
        if 'HTTPS_PROXY' in os.environ: os.environ.pop('HTTPS_PROXY')
        return

def objdump(obj, file='objdump.tmp'):
    import pickle
    with open(file, 'wb+') as f:
        pickle.dump(obj, f)
    return

def objload(file='objdump.tmp'):
    import pickle, os
    if not os.path.exists(file): 
        return
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def Singleton(cls):
    """
    単一のインスタンスデコレータ
    """
    _instance = {}
 
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]
 
    return _singleton

"""
========================================================================
テキストの翻訳
テキストの翻訳:
    - set_conf:                     実行中に設定を動的に変更する
    - set_multi_conf:               在运行过程中动态地修改多テキストの翻訳配置
    - get_plugin_handle:            プラグインのハンドルを取得する
    - get_plugin_default_kwargs:    テキストの翻訳
    - get_chat_handle:              テキストの翻訳
    - get_chat_default_kwargs:      テキストの翻訳
========================================================================
"""

def set_conf(key, value):
    from toolbox import read_single_conf_with_lru_cache, get_conf
    read_single_conf_with_lru_cache.cache_clear()
    get_conf.cache_clear()
    os.environ[key] = str(value)
    altered, = get_conf(key)
    return altered

def set_multi_conf(dic):
    for k, v in dic.items(): set_conf(k, v)
    return

def get_plugin_handle(plugin_name):
    """
    e.g. plugin_name = '原始文本>原始文本'
    """
    import importlib
    assert '->' in plugin_name, \
        "Example of plugin_name: 原始文本>原始文本"
    module, fn_name = plugin_name.split('->')
    f_hot_reload = getattr(importlib.import_module(module, fn_name), fn_name)
    return f_hot_reload

def get_chat_handle():
    """
    """
    from request_llm.bridge_all import predict_no_ui_long_connection
    return predict_no_ui_long_connection

def get_plugin_default_kwargs():
    """
    """
    from toolbox import get_conf, ChatBotWithCookies

    WEB_PORT, LLM_MODEL, API_KEY = \
        get_conf('WEB_PORT', 'LLM_MODEL', 'API_KEY')

    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p':1.0, 
        'max_length': None,
        'temperature':1.0,
    }
    chatbot = ChatBotWithCookies(llm_kwargs)

    # txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port
    DEFAULT_FN_GROUPS_kwargs = {
        "main_input": "./README.md",
        "llm_kwargs": llm_kwargs,
        "plugin_kwargs": {},
        "chatbot_with_cookie": chatbot,
        "history": [],
        "system_prompt": "You are a good AI.", 
        "web_port": WEB_PORT
    }
    return DEFAULT_FN_GROUPS_kwargs

def get_chat_default_kwargs():
    """
    """
    from toolbox import get_conf

    LLM_MODEL, API_KEY = get_conf('LLM_MODEL', 'API_KEY')

    llm_kwargs = {
        'api_key': API_KEY,
        'llm_model': LLM_MODEL,
        'top_p':1.0, 
        'max_length': None,
        'temperature':1.0,
    }

    default_chat_kwargs = {
        "inputs": "Hello there, are you ready?",
        "llm_kwargs": llm_kwargs,
        "history": [],
        "sys_prompt": "You are AI assistant",
        "observe_window": None,
        "console_slience": False,
    }

    return default_chat_kwargs

