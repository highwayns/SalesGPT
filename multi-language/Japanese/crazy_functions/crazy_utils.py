from toolbox import update_ui, get_conf, trimmed_format_exc, get_log_folder
import threading
import os
import logging

def input_clipping(inputs, history, max_token_limit):
    import numpy as np
    from request_llm.bridge_all import model_info
    enc = model_info["gpt-3.5-turbo"]['tokenizer']
    def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))

    mode = 'input-and-history'
    # 入力部分のトークンの割合が全体の半分未満の場合，過去のみをトリミングする
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit//2: 
        mode = 'only-history'
        max_token_limit = max_token_limit - input_token_num

    everything = [inputs] if mode == 'input-and-history' else ['']
    everything.extend(history)
    n_token = get_token_num('\n'.join(everything))
    everything_token = [get_token_num(e) for e in everything]
    delta = max(everything_token) // 16 # 切り捨て時の粒度
        
    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = enc.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]
        everything[where] = enc.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))

    if mode == 'input-and-history':
        inputs = everything[0]
    else:
        pass
    history = everything[1:]
    return inputs, history

def request_gpt_model_in_new_thread_with_ui_alive(
        inputs, inputs_show_user, llm_kwargs, 
        chatbot, history, sys_prompt, refresh_interval=0.2,
        handle_token_exceed=True, 
        retry_times_at_unknown_error=2,
        ):
    """
    Request GPT model，GPTモデルのリクエストとユーザーインターフェースの活性化を同時に維持する。

    引数を入力してください （_arrayで終わる入力変数はすべてリストです，リストの長さはサブタスクの数です，実行時，リストを分解します，それぞれのサブスレッドに配置して実行する）:
        inputs (string): List of inputs （输入）
        inputs_show_user (string): List of inputs to show user（テキストの翻訳，このパラメータを利用する，冗長な実際の入力を集計レポートで非表示にします，レポートの可読性を向上させる）
        top_p (float): Top p value for sampling from model distribution （原始文本，原始文本）
        temperature (float): Temperature value for sampling from model distribution（原始文本，原始文本）
        chatbot: chatbot inputs and outputs （ユーザーインターフェースのダイアログウィンドウハンドル，データフローの可視化に使用するされます）
        history (list): List of chat history （テキストの翻訳，対話履歴リスト）
        sys_prompt (string): List of system prompts （システム入力，リスト，用于输入给GPT的前提テキストの翻訳，たとえば、あなたが通訳である場合はどうなりますか）
        refresh_interval (float, optional): Refresh interval for UI (default: 0.2) （リフレッシュ時間間隔の頻度，アドバイス低于1，3を超えることはできません，視覚効果のためにのみサービスを提供）
        handle_token_exceed：トークンのオーバーフローを自動的に処理するかどうか，自動処理を選択する場合，原始文本，デフォルトで有効にする
        retry_times_at_unknown_error：失敗時のリトライ回数

    出力 Returns:
        future: 输出，GPTの結果を返す
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from request_llm.bridge_all import predict_no_ui_long_connection
    # ユーザーフィードバック
    chatbot.append([inputs_show_user, ""])
    yield from update_ui(chatbot=chatbot, history=[]) # 画面をリフレッシュする
    executor = ThreadPoolExecutor(max_workers=16)
    mutable = ["", time.time(), ""]
    # ウォッチドッグの忍耐力
    watch_dog_patience = 5
    # タスクを要求する
    def _req_gpt(inputs, history, sys_prompt):
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        while True:
            # watchdog error
            if len(mutable) >= 2 and (time.time()-mutable[1]) > watch_dog_patience: 
                raise RuntimeError("プログラムの終了が検出されました。")
            try:
                # 【原始文本】：成功裏に完了する
                result = predict_no_ui_long_connection(
                    inputs=inputs, llm_kwargs=llm_kwargs,
                    history=history, sys_prompt=sys_prompt, observe_window=mutable)
                return result
            except ConnectionAbortedError as token_exceeded_error:
                # 【テキストの翻訳】：トークンのオーバーフロー
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # 【処理の選択】 比率の計算を試みる，テキストをできるだけ多く保持します
                    from toolbox import get_reduce_token_percent
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    mutable[0] += f'[Local Message] 警告，テキストの翻訳，トークンのオーバーフロー数：{n_exceed}。\n\n'
                    continue # テキストの翻訳
                else:
                    # 【放棄を選択】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    mutable[0] += f"[Local Message] 警告，問題が発生しました, Traceback：\n\n{tb_str}\n\n"
                    return mutable[0] # 放棄
            except:
                # 【3番目の場合】：その他のエラー：テキストの翻訳
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print(tb_str)
                mutable[0] += f"[Local Message] 警告，問題が発生しました, Traceback：\n\n{tb_str}\n\n"
                if retry_op > 0:
                    retry_op -= 1
                    mutable[0] += f"[Local Message] 再試行中，少々お待ちください {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}：\n\n"
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        time.sleep(30)
                    time.sleep(5)
                    continue # テキストの翻訳
                else:
                    time.sleep(5)
                    return mutable[0] # 放棄

    # タスクの提出
    future = executor.submit(_req_gpt, inputs, history, sys_prompt)
    while True:
        # フロントエンドページを更新するために1回yieldする
        time.sleep(refresh_interval)
        # 「喂狗」（ウォッチドッグ）
        mutable[1] = time.time()
        if future.done():
            break
        chatbot[-1] = [chatbot[-1][0], mutable[0]]
        yield from update_ui(chatbot=chatbot, history=[]) # 画面をリフレッシュする

    final_result = future.result()
    chatbot[-1] = [chatbot[-1][0], final_result]
    yield from update_ui(chatbot=chatbot, history=[]) # 原始文本，原始文本
    return final_result

def can_multi_process(llm):
    if llm.startswith('gpt-'): return True
    if llm.startswith('api2d-'): return True
    if llm.startswith('azure-'): return True
    return False

def request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
        inputs_array, inputs_show_user_array, llm_kwargs, 
        chatbot, history_array, sys_prompt_array, 
        refresh_interval=0.2, max_workers=-1, scroller_max_len=30,
        handle_token_exceed=True, show_user_at_complete=False,
        retry_times_at_unknown_error=2,
        ):
    """
    Request GPT model using multiple threads with UI and high efficiency
    GPTモデルのリクエスト[マルチスレッド]版。
    テキストの翻訳：
        リモートデータストリームをUI上でリアルタイムにフィードバックする
        スレッドプールを使用するする，OpenAIのトラフィック制限エラーを回避するためにスレッドプールのサイズを調整できます
        途中で中止された処理
        ネットワークなどの問題が発生した場合，tracebackと受信済みのデータを出力に変換します

    引数を入力してください （_arrayで終わる入力変数はすべてリストです，リストの長さはサブタスクの数です，実行時，リストを分解します，それぞれのサブスレッドに配置して実行する）:
        inputs_array (list): List of inputs （各子タスクの入力）
        inputs_show_user_array (list): List of inputs to show user（各サブタスクの入力がレポートに表示されます，このパラメータを利用する，冗長な実際の入力を集計レポートで非表示にします，レポートの可読性を向上させる）
        llm_kwargs: llm_kwargsパラメータ
        chatbot: chatbot （ユーザーインターフェースのダイアログウィンドウハンドル，データフローの可視化に使用するされます）
        history_array (list): List of chat history （過去の対話の入力，テキストの翻訳，最初のリストはサブタスクの分解です，2番目のリストは会話履歴です）
        sys_prompt_array (list): List of system prompts （システム入力，リスト，用于输入给GPT的前提テキストの翻訳，たとえば、あなたが通訳である場合はどうなりますか）
        refresh_interval (float, optional): Refresh interval for UI (default: 0.2) （リフレッシュ時間間隔の頻度，アドバイス低于1，3を超えることはできません，視覚効果のためにのみサービスを提供）
        max_workers (int, optional): Maximum number of threads (default: see config.py) （最大スレッド数，サブタスクが非常に多い場合，原始文本）
        scroller_max_len (int, optional): Maximum length for scroller (default: 30)（数据流的显示原始文本收到的多少テキストの翻訳字符，視覚効果のためにのみサービスを提供）
        handle_token_exceed (bool, optional): （原始文本，テキストの翻訳）
        handle_token_exceed：トークンのオーバーフローを自動的に処理するかどうか，自動処理を選択する場合，原始文本，デフォルトで有効にする
        show_user_at_complete (bool, optional): (テキストの翻訳，完全な入力-出力結果をチャットボックスに表示する)
        retry_times_at_unknown_error：子タスクが失敗した場合の再試行回数

    出力 Returns:
        list: List of GPT model responses （テキストの翻訳，テキストの翻訳，responseにはtracebackエラーメッセージが含まれます，原始文本。）
    """
    import time, random
    from concurrent.futures import ThreadPoolExecutor
    from request_llm.bridge_all import predict_no_ui_long_connection
    assert len(inputs_array) == len(history_array)
    assert len(inputs_array) == len(sys_prompt_array)
    if max_workers == -1: # 設定ファイルの読み込み
        try: max_workers, = get_conf('DEFAULT_WORKER_NUM')
        except: max_workers = 8
        if max_workers <= 0: max_workers = 3
    # chatglmのマルチスレッドを無効にする，重度の遅延を引き起こす可能性があります
    if not can_multi_process(llm_kwargs['llm_model']):
        max_workers = 1
        
    executor = ThreadPoolExecutor(max_workers=max_workers)
    n_frag = len(inputs_array)
    # ユーザーフィードバック
    chatbot.append(["マルチスレッド操作を開始してください。", ""])
    yield from update_ui(chatbot=chatbot, history=[]) # 画面をリフレッシュする
    # テキストの翻訳
    mutable = [["", time.time(), "テキストの翻訳"] for _ in range(n_frag)]

    # ウォッチドッグの忍耐力
    watch_dog_patience = 5

    # サブスレッドのタスク
    def _req_gpt(index, inputs, history, sys_prompt):
        gpt_say = ""
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        mutable[index][2] = "执行中"
        while True:
            # watchdog error
            if len(mutable[index]) >= 2 and (time.time()-mutable[index][1]) > watch_dog_patience: 
                raise RuntimeError("プログラムの終了が検出されました。")
            try:
                # 【原始文本】：成功裏に完了する
                # time.sleep(10); raise RuntimeError("测试")
                gpt_say = predict_no_ui_long_connection(
                    inputs=inputs, llm_kwargs=llm_kwargs, history=history, 
                    sys_prompt=sys_prompt, observe_window=mutable[index], console_slience=True
                )
                mutable[index][2] = "成功しました"
                return gpt_say
            except ConnectionAbortedError as token_exceeded_error:
                # 【テキストの翻訳】：トークンのオーバーフロー，
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # 【処理の選択】 比率の計算を試みる，テキストをできるだけ多く保持します
                    from toolbox import get_reduce_token_percent
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    gpt_say += f'[Local Message] 警告，テキストの翻訳，トークンのオーバーフロー数：{n_exceed}。\n\n'
                    mutable[index][2] = f"切り捨て再試行"
                    continue # テキストの翻訳
                else:
                    # 【放棄を選択】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    gpt_say += f"[Local Message] 警告，テキストの翻訳{index}問題が発生しました, Traceback：\n\n{tb_str}\n\n"
                    if len(mutable[index][0]) > 0: gpt_say += "このスレッドが失敗する前に受け取った回テキストの翻訳：\n\n" + mutable[index][0]
                    mutable[index][2] = "入力が長すぎるため、放棄されました"
                    return gpt_say # 放棄
            except:
                # 【3番目の場合】：その他のエラー
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print(tb_str)
                gpt_say += f"[Local Message] 警告，テキストの翻訳{index}問題が発生しました, Traceback：\n\n{tb_str}\n\n"
                if len(mutable[index][0]) > 0: gpt_say += "このスレッドが失敗する前に受け取った回テキストの翻訳：\n\n" + mutable[index][0]
                if retry_op > 0: 
                    retry_op -= 1
                    wait = random.randint(5, 20)
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        wait = wait * 3
                        fail_info = "OpenAIはクレジットカードのバインドにより頻度制限を解除できます "
                    else:
                        fail_info = ""
                    # おそらく数十秒待つと，情况会好转
                    for i in range(wait):
                        mutable[index][2] = f"{fail_info}再試行を待っています {wait-i}"; time.sleep(1)
                    # 开始重试
                    mutable[index][2] = f"再試行中 {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}"
                    continue # テキストの翻訳
                else:
                    mutable[index][2] = "テキストの翻訳"
                    wait = 5
                    time.sleep(5)
                    return gpt_say # 放棄

    # 非同期タスクの開始
    futures = [executor.submit(_req_gpt, index, inputs, history, sys_prompt) for index, inputs, history, sys_prompt in zip(
        range(len(inputs_array)), inputs_array, history_array, sys_prompt_array)]
    cnt = 0
    while True:
        # フロントエンドページを更新するために1回yieldする
        time.sleep(refresh_interval)
        cnt += 1
        worker_done = [h.done() for h in futures]
        # より良いUIの視覚効果
        observe_win = []
        # 各スレッドは「犬にエサをやる」必要があります（ウォッチドッグ）
        for thread_index, _ in enumerate(worker_done):
            mutable[thread_index][1] = time.time()
        # フロントエンドで面白いものを印刷する
        for thread_index, _ in enumerate(worker_done):
            print_something_really_funny = "[ ...`"+mutable[thread_index][0][-scroller_max_len:].\
                replace('\n', '').replace('`', '.').replace(
                    ' ', '.').replace('<br/>', '.....').replace('$', '.')+"`... ]"
            observe_win.append(print_something_really_funny)
        # フロントエンドで面白いものを印刷する
        stat_str = ''.join([f'`{mutable[thread_index][2]}`: {obs}\n\n' 
                            if not done else f'`{mutable[thread_index][2]}`\n\n' 
                            for thread_index, done, obs in zip(range(len(worker_done)), worker_done, observe_win)])
        # フロントエンドで面白いものを印刷する
        chatbot[-1] = [chatbot[-1][0], f'テキストの翻訳，完了状況: \n\n{stat_str}' + ''.join(['.']*(cnt % 10+1))]
        yield from update_ui(chatbot=chatbot, history=[]) # 画面をリフレッシュする
        if all(worker_done):
            executor.shutdown()
            break

    # 异步任务終了
    gpt_response_collection = []
    for inputs_show_user, f in zip(inputs_show_user_array, futures):
        gpt_res = f.result()
        gpt_response_collection.extend([inputs_show_user, gpt_res])
    
    # テキストの翻訳，結果を画面に表示する
    if show_user_at_complete:
        for inputs_show_user, f in zip(inputs_show_user_array, futures):
            gpt_res = f.result()
            chatbot.append([inputs_show_user, gpt_res])
            yield from update_ui(chatbot=chatbot, history=[]) # 画面をリフレッシュする
            time.sleep(0.5)
    return gpt_response_collection


def breakdown_txt_to_satisfy_token_limit(txt, get_token_fn, limit):
    def cut(txt_tocut, must_break_at_empty_line):  # 递归
        if get_token_fn(txt_tocut) <= limit:
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                print(cnt)
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break
            if cnt == 0:
                raise RuntimeError("非常に長いテキストが1行に存在します！")
            # print(len(post))
            # リストの再帰的な連鎖
            result = [prev]
            result.extend(cut(post, must_break_at_empty_line))
            return result
    try:
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        return cut(txt, must_break_at_empty_line=False)


def force_breakdown(txt, limit, get_token_fn):
    """
    句読点や空行で分割できない場合，私たち用最暴力的テキストの翻訳切割
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktokenの未知のエラー", "Tiktokenの未知のエラー"

def breakdown_txt_to_satisfy_token_limit_for_pdf(txt, get_token_fn, limit):
    # 递归
    def cut(txt_tocut, must_break_at_empty_line, break_anyway=False):  
        if get_token_fn(txt_tocut) <= limit:
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break
            if cnt == 0:
                if break_anyway:
                    prev, post = force_breakdown(txt_tocut, limit, get_token_fn)
                else:
                    raise RuntimeError(f"非常に長いテキストが1行に存在します！{txt_tocut}")
            # print(len(post))
            # リストの再帰的な連鎖
            result = [prev]
            result.extend(cut(post, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # テキストの翻訳，二重の空行を（\n\n）分割点として
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        try:
            # 2回目の試み，テキストの翻訳（\n）分割点として
            return cut(txt, must_break_at_empty_line=False)
        except RuntimeError:
            try:
                # 第3回目の試み，英文の句点を（.）分割点として
                res = cut(txt.replace('.', '。\n'), must_break_at_empty_line=False) # この中国語の句点は意図的です，識別子として存在します
                return [r.replace('。\n', '.') for r in res]
            except RuntimeError as e:
                try:
                    # 4回目の試み，中国語の句点を（。）分割点として
                    res = cut(txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                    return [r.replace('。。\n', '。') for r in res]
                except RuntimeError as e:
                    # 5回目の試み，テキストの翻訳，原始文本
                    return cut(txt, must_break_at_empty_line=False, break_anyway=True)



def read_and_clean_pdf_text(fp):
    """
    テキストの翻訳，原始文本，原始文本，効果が非常に良い

    **入力パラメータの説明**
    - `fp`：テキストを読み取り、クリーンアップするためのPDFファイルのパスが必要です

    **出力パラメータの説明**
    - `meta_txt`：クリーンアップされたテキストコンテンツ文字列
    - `page_one_meta`：クリーンアップ後のテキストコンテンツリストの最初のページ

    **関数の機能**
    pdfファイルを読み取り、テキスト内容をクリーンアップする，クリーニングルールには以下が含まれます：
    - すべてのブロック要素のテキスト情報を抽出する，結合して1つの文字列にする
    - 短いブロックを削除する（文字数が100未満）テキストの翻訳
    - cleanUpExcessEmptyLines
    - 小文字で始まる段落ブロックを結合してスペースに置換する
    - 重複した改行を削除
    - テキストの翻訳，テキストの翻訳
    """
    import fitz, copy
    import re
    import numpy as np
    from colorful import printBrightYellow, printBrightGreen
    fc = 0  # テキストの翻訳
    fs = 1  # インデックス1のフォント
    fb = 2  # インデックス2 フレーム
    REMOVE_FOOT_NOTE = True # 本文でないコンテンツを破棄するかどうか （テキストの翻訳，参考文献、脚注、図表のようなもの）
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95 # 本文より小さい？時，本文ではないと判断される（一部の記事の本文のフォントサイズは100％統一されていません，肉眼では見えない微細な変化があります）
    def primary_ffsize(l):
        """
        テキストブロックのメインフォントを抽出する
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs: fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)
        
    def ffsize_same(a,b):
        """
        原始文本
        """
        return abs((a-b)/max(a,b)) < 0.02

    with fitz.open(fp) as doc:
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        ############################## <テキストの翻訳，原始文本> ##################################
        for index, page in enumerate(doc):
            # file_content += page.get_text()
            text_areas = page.get_text("dict")  # ページ上のテキスト情報を取得する
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0: continue
                        pf = primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']: # for l in t['lines']:
                            meta_span.append([wtf['text'], wtf['size'], len(wtf['text'])])
                    # meta_line.append(["NEW_BLOCK", pf])
            # テキストの翻訳                           for each word segment with in line                       for each line         cross-line words                          for each block
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                             for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]
                
        ############################## <ステップ2，原始文本> ##################################
        try:
            fsize_statiscs = {}
            for span in meta_span:
                if span[1] not in fsize_statiscs: fsize_statiscs[span[1]] = 0
                fsize_statiscs[span[1]] += span[2]
            main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
            if REMOVE_FOOT_NOTE:
                give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT
        except:
            raise RuntimeError(f'申し訳ありません, このPDFドキュメントを解析することはできません: {fp}。')
        ############################## <ステップ 3，分割と再結合> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0: 
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if ffsize_same(meta_line[index][fs], meta_line[index-1][fs]):
                # テキストの翻訳
                if meta_line[index][fc].endswith('.') and\
                    (meta_line[index-1][fc] != 'NEW_BLOCK') and \
                    (meta_line[index][fb][2] - meta_line[index][fb][0]) < (meta_line[index-1][fb][2] - meta_line[index-1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index+1 < len(meta_line)) and \
                    meta_line[index][fs] > main_fsize:
                    # 単一行 + 大文字フォント
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # セクションを識別しようとします
                    if meta_line[index-1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <第4ステップ，乱雑な後処理> ##################################
        def clearBlocksWithTooFewCharactersToNewLine(meta_txt):
            for index, block_txt in enumerate(meta_txt):
                if len(block_txt) < 100:
                    meta_txt[index] = '\n'
            return meta_txt
        meta_txt = clearBlocksWithTooFewCharactersToNewLine(meta_txt)

        def cleanUpExcessEmptyLines(meta_txt):
            for index in reversed(range(1, len(meta_txt))):
                if meta_txt[index] == '\n' and meta_txt[index-1] == '\n':
                    meta_txt.pop(index)
            return meta_txt
        meta_txt = cleanUpExcessEmptyLines(meta_txt)

        def mergeParagraphBlocksStartingWithLowerCase(meta_txt):
            def starts_with_lowercase_word(s):
                pattern = r"^[a-z]+"
                match = re.match(pattern, s)
                if match:
                    return True
                else:
                    return False
            for _ in range(100):
                for index, block_txt in enumerate(meta_txt):
                    if starts_with_lowercase_word(block_txt):
                        if meta_txt[index-1] != '\n':
                            meta_txt[index-1] += ' '
                        else:
                            meta_txt[index-1] = ''
                        meta_txt[index-1] += meta_txt[index]
                        meta_txt[index] = '\n'
            return meta_txt
        meta_txt = mergeParagraphBlocksStartingWithLowerCase(meta_txt)
        meta_txt = cleanUpExcessEmptyLines(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # 重複した改行を削除
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # 改行 -> 二重改行
        meta_txt = meta_txt.replace('\n', '\n\n')

        ############################## <ステップ5，分割効果を表示する> ##################################
        # for f in finals:
        #    printBrightYellow(f)
        #    printBrightGreen('***************************')

    return meta_txt, page_one_meta


def get_files_from_everything(txt, type): # type='.md'
    """
    テキストの翻訳（例えば.md）テキストの翻訳，そして、インターネット上のファイルに対して，也可以获取它。
    テキストの翻訳：
    テキストの翻訳 
    - txt: パスまたはURL，原始文本。 
    - type: 文字列，検索するファイルのタイプを表示する。デフォルトは.mdです。
    戻り値 
    - success: ブール値，関数の実行が成功したかどうかを示す。 
    - file_manifest: ファイルパスのリスト，指定された拡張子を持つすべてのファイルの絶対パスを含みます。 
    - project_folder: 文字列，ファイルが存在するフォルダのパスを表示する。ネット上のファイルの場合，一時フォルダのパスです。
    テキストの翻訳，要件を満たしているかどうかを確認してください。
    """
    import glob, os

    success = True
    if txt.startswith('http'):
        # ネットワークのリモートファイル
        import requests
        from toolbox import get_conf
        from toolbox import get_log_folder, gen_time_str
        proxies, = get_conf('proxies')
        try:
            r = requests.get(txt, proxies=proxies)
        except:
            raise ConnectionRefusedError(f"リソースをダウンロードできません{txt}，確認してください。")
        path = os.path.join(get_log_folder(plugin_name='web_download'), gen_time_str()+type)
        with open(path, 'wb+') as f: f.write(r.content)
        project_folder = get_log_folder(plugin_name='web_download')
        file_manifest = [path]
    elif txt.endswith(type):
        # 直接ファイルを指定する
        file_manifest = [txt]
        project_folder = os.path.dirname(txt)
    elif os.path.exists(txt):
        # テキストの翻訳，テキストの翻訳
        project_folder = txt
        file_manifest = [f for f in glob.glob(f'{project_folder}/**/*'+type, recursive=True)]
        if len(file_manifest) == 0:
            success = False
    else:
        project_folder = None
        file_manifest = []
        success = False

    return success, file_manifest, project_folder




def Singleton(cls):
    _instance = {}
 
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]
 
    return _singleton


@Singleton
class knowledge_archive_interface():
    def __init__(self) -> None:
        self.threadLock = threading.Lock()
        self.current_id = ""
        self.kai_path = None
        self.qa_handle = None
        self.text2vec_large_chinese = None

    def get_chinese_text2vec(self):
        if self.text2vec_large_chinese is None:
            # < -------------------プレヒートテキストベクトル化モジュール--------------- >
            from toolbox import ProxyNetworkActivate
            print('Checking Text2vec ...')
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings
            with ProxyNetworkActivate('Download_LLM'):    # 一時的にプロキシネットワークをアクティブ化します
                self.text2vec_large_chinese = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

        return self.text2vec_large_chinese


    def feed_archive(self, file_manifest, id="default"):
        self.threadLock.acquire()
        # import uuid
        self.current_id = id
        from zh_langchain import construct_vector_store
        self.qa_handle, self.kai_path = construct_vector_store(   
            vs_id=self.current_id, 
            files=file_manifest, 
            sentence_size=100,
            history=[],
            one_conent="",
            one_content_segmentation="",
            text2vec = self.get_chinese_text2vec(),
        )
        self.threadLock.release()

    def get_current_archive_id(self):
        return self.current_id
    
    def get_loaded_file(self):
        return self.qa_handle.get_loaded_file()

    def answer_with_archive_by_id(self, txt, id):
        self.threadLock.acquire()
        if not self.current_id == id:
            self.current_id = id
            from zh_langchain import construct_vector_store
            self.qa_handle, self.kai_path = construct_vector_store(   
                vs_id=self.current_id, 
                files=[], 
                sentence_size=100,
                history=[],
                one_conent="",
                one_content_segmentation="",
                text2vec = self.get_chinese_text2vec(),
            )
        VECTOR_SEARCH_SCORE_THRESHOLD = 0
        VECTOR_SEARCH_TOP_K = 4
        CHUNK_SIZE = 512
        resp, prompt = self.qa_handle.get_knowledge_based_conent_test(
            query = txt,
            vs_path = self.kai_path,
            score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
            vector_search_top_k=VECTOR_SEARCH_TOP_K, 
            chunk_conent=True,
            chunk_size=CHUNK_SIZE,
            text2vec = self.get_chinese_text2vec(),
        )
        self.threadLock.release()
        return resp, prompt
    
@Singleton
class nougat_interface():
    def __init__(self):
        self.threadLock = threading.Lock()

    def nougat_with_timeout(self, command, cwd, timeout=3600):
        import subprocess
        logging.info(f'コマンドを実行中 {command}')
        process = subprocess.Popen(command, shell=True, cwd=cwd)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print("Process timed out!")
            return False
        return True


    def NOUGAT_parse_pdf(self, fp, chatbot, history):
        from toolbox import update_ui_lastest_msg

        yield from update_ui_lastest_msg("論文を解析中です, お待ちください。進捗：キューに並んでいます, スレッドロックを待っています...", 
                                         chatbot=chatbot, history=history, delay=0)
        self.threadLock.acquire()
        import glob, threading, os
        from toolbox import get_log_folder, gen_time_str
        dst = os.path.join(get_log_folder(plugin_name='nougat'), gen_time_str())
        os.makedirs(dst)

        yield from update_ui_lastest_msg("論文を解析中です, お待ちください。進捗：NOUGATをロードしています... （テキストの翻訳：初回実行にはNOUGATパラメータのダウンロードにかなりの時間がかかります）", 
                                         chatbot=chatbot, history=history, delay=0)
        self.nougat_with_timeout(f'nougat --out "{os.path.abspath(dst)}" "{os.path.abspath(fp)}"', os.getcwd(), timeout=3600)
        res = glob.glob(os.path.join(dst,'*.mmd'))
        if len(res) == 0:
            self.threadLock.release()
            raise RuntimeError("Nougatの論文解析に失敗しました。")
        self.threadLock.release()
        return res[0]




def try_install_deps(deps, reload_m=[]):
    import subprocess, sys, importlib
    for dep in deps:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', dep])
    import site
    importlib.reload(site)
    for m in reload_m:
        importlib.reload(__import__(m))


HTML_CSS = """
.row {
  display: flex;
  flex-wrap: wrap;
}
.column {
  flex: 1;
  padding: 10px;
}
.table-header {
  font-weight: bold;
  border-bottom: 1px solid black;
}
.table-row {
  border-bottom: 1px solid lightgray;
}
.table-cell {
  padding: 5px;
}
"""

TABLE_CSS = """
<div class="row table-row">
    <div class="column table-cell">REPLACE_A</div>
    <div class="column table-cell">REPLACE_B</div>
</div>
"""

class construct_html():
    def __init__(self) -> None:
        self.css = HTML_CSS
        self.html_string = f'<!DOCTYPE html><head><meta charset="utf-8"><title>翻訳結果</title><style>{self.css}</style></head>'


    def add_row(self, a, b):
        tmp = TABLE_CSS
        from toolbox import markdown_convertion
        tmp = tmp.replace('REPLACE_A', markdown_convertion(a))
        tmp = tmp.replace('REPLACE_B', markdown_convertion(b))
        self.html_string += tmp


    def save_file(self, file_name):
        with open(os.path.join(get_log_folder(), file_name), 'w', encoding='utf8') as f:
            f.write(self.html_string.encode('utf-8', 'ignore').decode())
        return os.path.join(get_log_folder(), file_name)


def get_plugin_arg(plugin_kwargs, key, default):
    # 原始文本
    if (key in plugin_kwargs) and (plugin_kwargs[key] == ""): plugin_kwargs.pop(key)
    # 通常の状況では
    return plugin_kwargs.get(key, default)
