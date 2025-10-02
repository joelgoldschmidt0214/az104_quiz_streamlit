"""Streamlit app.py"""

import json

import streamlit as st
from google import genai
from pydantic import BaseModel, Field

# Geminiモデルを初期化
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash-lite"

QUIZ_NUM = 10


# --- プロンプトテンプレート ---
# プロンプトを工夫することで、出力の質をコントロールできます。

AZ104_CATEGORIES = [
    "IDとガバナンスの管理",
    "ストレージの実装と管理",
    "コンピューティングリソースのデプロイと管理",
    "仮想ネットワークの実装と管理",
    "Azureリソースの監視とバックアップ",
]


class OptionModel(BaseModel):
    id: str = Field(..., description="選択肢のID(例: 'A', 'B')")
    text: str = Field(..., description="選択肢の本文")


class QuizModel(BaseModel):
    question: str
    options: list[OptionModel]
    answer: list[str]
    type: str
    category: str


def create_prompt(selected_categories):
    # 選択されたカテゴリがなければ、全カテゴリからランダムに選ぶ
    if not selected_categories:
        categories_text = "Azure全般(特に以下のカテゴリ):\n" + "\n".join(f"- {cat}" for cat in AZ104_CATEGORIES)
    else:
        categories_text = "以下のカテゴリに焦点を当ててください:\n" + "\n".join(
            f"- {cat}" for cat in selected_categories
        )

    return f"""
    あなたはAzureの専門家です。AZ-104の模擬試験問題を作成してください。
    以下の条件に従って、日本語で{QUIZ_NUM}問の問題と選択肢、正解、問題形式をJSON形式で生成してください。

    # 条件
    - 難易度: AZ-104と同等か、少し難しいレベル
    - 出題範囲: {categories_text}
    - 問題形式: "single"(単一回答)または "multiple"(複数回答)をランダムに含めること
    - 出力形式: JSONリスト(配列)形式。前後に説明文や```jsonのようなマークダウンは含めないこと。

    [
      {{
        "question": "問題文",
        "options": [
            {{"id": "A", "text": "選択肢A"}},
            {{"id": "B", "text": "選択肢B"}},
            {{"id": "C", "text": "選択肢C"}},
            {{"id": "D", "text": "選択肢D"}}
        ],
        "answer": ["正解のid"],
        "type": "single",
        "category": "出題カテゴリ名"
      }},
      ...
    ]
    """


# --- Streamlit アプリケーション ---

st.title("Azure AZ-104 模擬試験アプリ")

# セッション状態で問題と回答を管理
if "questions" not in st.session_state:
    st.session_state.questions = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# サイドバーでカテゴリ選択(Nice-to-have機能)
st.sidebar.header("出題カテゴリ")
selected_cats = st.sidebar.multiselect(
    "カテゴリを選択してください(未選択の場合はランダム)",
    AZ104_CATEGORIES,
)

if st.sidebar.button(f"新しい問題を{QUIZ_NUM}問生成", key="generate_button"):
    st.session_state.submitted = False
    with st.spinner("Geminiが問題を生成中です..."):
        prompt = create_prompt(selected_cats)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[QuizModel],
                },
            )
            st.session_state.questions = response.parsed
            st.session_state.user_answers = {}  # ユーザーの回答をリセット
            st.rerun()  # 画面を再描画して問題を表示
        except Exception as e:
            st.error(f"問題の生成中にエラーが発生しました: {e}")
            st.error(f"受信したデータ: {response.parsed}")  # デバッグ用に表示

# 問題が生成されていれば表示
if st.session_state.questions and not st.session_state.submitted:
    st.write("問題取得完了")
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.questions):
            st.subheader(f"問題 {i + 1}")
            st.write(q.question)

            # options = list(q["options"].keys())
            options_ids = [opt.id for opt in q.options]
            options_dict = {opt.id: opt.text for opt in q.options}

            if q.type == "single":
                answer = st.radio(
                    "選択肢:",
                    options_ids,
                    key=f"q_{i}",
                    format_func=lambda x, cur_dict=options_dict: f"{x}: {cur_dict[x]}",
                )
                st.session_state.user_answers[i] = [answer]
            elif q.type == "multiple":
                answers = st.multiselect(
                    "選択肢 (複数選択可):",
                    options_ids,
                    key=f"q_{i}",
                    format_func=lambda x, cur_dict=options_dict: f"{x}: {cur_dict[x]}",
                )
                st.session_state.user_answers[i] = answers

            # 任意記入欄
            st.text_area("この回答を選んだ理由(任意):", key=f"reason_{i}")

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
        correct_ans = sorted(q.answer)
        is_correct = user_ans == correct_ans

        st.subheader(f"問題 {i + 1}")
        st.write(q.question)

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
            with st.spinner(f"問題{i + 1}の解説を生成中..."):
                options_for_prompt = {opt.id: opt.text for opt in q.options}

                explanation_prompt = f"""
                以下のAzureの問題について、なぜこれが正解なのか解説してください。
                ユーザーの回答や記入内容も踏まえて、特に間違っている点を重点的に説明してください。

                # 問題
                {q.question}




                # 選択肢
                {json.dumps(options_for_prompt, ensure_ascii=False)}

                # 正解
                {", ".join(correct_ans)}

                # ユーザーの回答
                {", ".join(user_ans)}

                # ユーザーが記入した理由
                {reason if reason else "記載なし"}
                """
                try:
                    explanation_response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=explanation_prompt,
                    )
                    st.info(f"解説:\n{explanation_response.text}")
                except Exception as e:
                    st.warning(f"解説の生成に失敗しました: {e}")

    st.header(f"正答率: {correct_count / len(st.session_state.questions):.0%}")
