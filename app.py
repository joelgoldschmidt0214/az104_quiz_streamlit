import streamlit as st
from google import genai
import json
import random


# Geminiモデルを初期化
client = genai.Client()
model = 'gemini-2.5-flash'


# APIキーを設定
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# --- プロンプトテンプレート ---
# プロンプトを工夫することで、出力の質をコントロールできます。
PROMPT_TEMPLATE = """
あなたはAzureの専門家です。AZ-104の模擬試験問題を作成してください。
以下の条件に従って、日本語で10問の問題と選択肢、正解、問題形式を絶対にJSON形式で生成してください。
JSON以外の文字列は一切出力しないでください。

# 条件
- 難易度: AZ-104と同等か、少し難しいレベル
- 問題形式: "single"（単一回答）または "multiple"（複数回答）をランダムに含める
- 出力形式: 以下のJSON形式のリスト（配列）とすること

[
  {
    "question": "問題文をここに記述",
    "options": {
      "A": "選択肢A",
      "B": "選択肢B",
      "C": "選択肢C",
      "D": "選択肢D"
    },
    "answer": ["正解のキー"],
    "type": "single"
  },
  {
    "question": "問題文をここに記述",
    "options": {
      "A": "選択肢A",
      "B": "選択肢B",
      "C": "選択肢C",
      "D": "選択肢D"
    },
    "answer": ["正解のキー1", "正解のキー2"],
    "type": "multiple"
  }
]
"""

# --- Streamlit アプリケーション ---

st.title("Azure AZ-104 模擬試験アプリ")

# セッション状態で問題と回答を管理
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if st.button("新しい問題を10問生成", key="generate_button"):
    st.session_state.submitted = False
    with st.spinner("Geminiが問題を生成中です..."):
        try:
            response = client.models.generate_content(model=model, contents=PROMPT_TEMPLATE)
            # Geminiからの応答がMarkdown形式のコードブロックを含む場合があるため、それを除去
            cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
            st.session_state.questions = json.loads(cleaned_response)
            st.session_state.user_answers = {} # ユーザーの回答をリセット
            st.rerun() # 画面を再描画して問題を表示
        except Exception as e:
            st.error(f"問題の生成中にエラーが発生しました: {e}")
            st.error(f"受信したデータ: {response.text}") # デバッグ用に表示

# 問題が生成されていれば表示
if st.session_state.questions and not st.session_state.submitted:
    st.write("問題取得完了")
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.questions):
            st.subheader(f"問題 {i+1}")
            st.write(q['question'])
            
            options = list(q['options'].keys())
            if q['type'] == 'single':
                answer = st.radio("選択肢:", options, key=f"q_{i}", format_func=lambda x: f"{x}: {q['options'][x]}")
                st.session_state.user_answers[i] = [answer]
            elif q['type'] == 'multiple':
                answers = st.multiselect("選択肢 (複数選択可):", options, key=f"q_{i}", format_func=lambda x: f"{x}: {q['options'][x]}")
                st.session_state.user_answers[i] = answers

            # 任意記入欄
            st.text_area("この回答を選んだ理由（任意）:", key=f"reason_{i}")

        submitted = st.form_submit_button("回答")
        if submitted:
            st.session_state.submitted = True
            st.rerun()

# 回答が送信されたら結果を表示
if st.session_state.submitted:
    st.header("結果発表")
    correct_count = 0
    
    for i, q in enumerate(st.session_state.questions):
        user_ans = sorted(st.session_state.user_answers.get(i, []))
        correct_ans = sorted(q['answer'])
        is_correct = (user_ans == correct_ans)
        
        st.subheader(f"問題 {i+1}")
        st.write(q['question'])
        
        if is_correct:
            st.success("正解！")
            correct_count += 1
        else:
            st.error("不正解")
        
        st.write(f"あなたの回答: {', '.join(user_ans)}")
        st.write(f"正解: {', '.join(correct_ans)}")

        # 解説生成
        reason = st.session_state.get(f"reason_{i}", "")
        if not is_correct or reason:
            with st.spinner(f"問題{i+1}の解説を生成中..."):
                explanation_prompt = f"""
                以下のAzureの問題について、なぜこれが正解なのか解説してください。
                ユーザーの回答や記入内容も踏まえて、特に間違っている点を重点的に説明してください。

                # 問題
                {q['question']}

                # 選択肢
                {json.dumps(q['options'], ensure_ascii=False)}

                # 正解
                {', '.join(correct_ans)}

                # ユーザーの回答
                {', '.join(user_ans)}

                # ユーザーが記入した理由
                {reason if reason else "記載なし"}
                """
                try:
                    explanation_response = client.models.generate_content(model=model, contents=explanation_prompt)
                    st.info(f"解説:\n{explanation_response.text}")
                except Exception as e:
                    st.warning(f"解説の生成に失敗しました: {e}")

    st.header(f"正答率: {correct_count / len(st.session_state.questions):.0%}")