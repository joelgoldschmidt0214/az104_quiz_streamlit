"""Streamlit app.py"""

import json

import streamlit as st
from google import genai
from pydantic import BaseModel, Field

st.set_page_config(page_title="AZ-104 Quiz", layout="wide")


# Geminiモデルを初期化
client = genai.Client()
# MODEL_NAME = "gemini-2.5-flash-lite"

QUIZ_NUM = 10

model_dict = {
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}


# --- プロンプトテンプレート ---
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


# --- Pydanticモデル定義 (ExplanationModelをこちらに差し替え) ---
class DetailedExplanationModel(BaseModel):
    question_index: int = Field(..., description="解説対象の問題のインデックス番号(0始まり)")
    overall_summary: str = Field(
        ...,
        description="この問題で問われているAzureの中核的な知識や概念についての要約（1〜2文）",
    )
    correct_answer_explanation: str = Field(..., description="正解の選択肢がなぜ正しいのかについての詳細な説明")
    user_feedback: str = Field(
        ...,
        description="ユーザーの回答がなぜ間違っていたのか、または考え方のどこを修正すべきかについての具体的でパーソナルなフィードバック。ユーザーが正解している場合は、その思考プロセスを称賛する内容。",
    )
    analysis_of_other_options: str = Field(
        ...,
        description="正解以外の主要な選択肢が、なぜ不正解なのかについての簡潔な解説",
    )


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


# --- プロンプトテンプレート (create_batch_explanation_prompt関数を更新) ---
def create_batch_explanation_prompt(indices_to_explain, all_questions, all_results):
    """選択された問題の詳細な解説を一度に生成するためのプロンプトを作成する"""

    questions_for_prompt = []
    for i in indices_to_explain:
        q = all_questions[i]
        result = all_results[i]
        reason = st.session_state.get(f"reason_{i}", "記載なし")

        questions_for_prompt.append(
            {
                "question_index": i,
                "question": q.question,
                "options": {opt.id: opt.text for opt in q.options},
                "correct_answer": q.answer,
                "user_answer": result["user_ans"],
                "reason_by_user": reason,
            },
        )

    prompt_header = """
    あなたは、非常に共感能力が高く、経験豊富なAzureの個別指導教師です。
    以下のJSONデータに含まれる複数の問題について、学習者の学習効果が最大になるように、詳細な解説を生成してください。

    # 全体的な指示
    - 生成する解説は、常に学習者である「あなた」に直接、優しく語りかける口調で記述してください。
      学習者のことを指して「ユーザー」や「受験者」という三人称の呼びかけは絶対に使わないでください。
      Azureサービスの「ユーザー」を表現する際は、「ユーザー」という単語を使っても構いません。
    - 最終的な出力は、指示されたJSONスキーマに厳密に従ったJSONリスト形式とし、
      前後に説明文やマークダウンは含めないでください。

    # JSONスキーマの各フィールドの詳細な指示
    - `question_index`: 元の質問のインデックス番号。
    - `overall_summary`: この問題が試しているAzureの核心的な知識を、あなたが理解できるよう簡潔にまとめたもの。
    - `correct_answer_explanation`: 正解の選択肢がなぜ正しいのかを、論理的かつ体系的に説明したもの。
    - `user_feedback`: ここが最も重要です。あなたの回答と、あなたが記入した理由(`reason_by_user`)を深く考慮した、
      パーソナルなフィードバックを生成してください。
        - もし `reason_by_user` に具体的な記述がある場合、必ずその内容に触れてください。
          例えば、「『[reason_by_userの内容]』という理由でこの選択肢を選ばれたのですね。
          その視点は素晴らしいですが、このシナリオでは〇〇という点が異なります。」のように、
          あなたの考えを一度受け止めてから、解説を始めてください。
        - あなたがなぜその間違いを犯したのかを分析し、次に同様の問題を解くための具体的なヒントや
          思考のコツを提示してください。
        - 正解していた場合でも、あなたの考え方を称賛し、さらに知識を深めるための補足情報を提供してください。
    - `analysis_of_other_options`: あなたが誤答した、あるいは間違いである他の選択肢について簡潔に説明したもの。
    """

    return f"""
    {prompt_header}

    # 解説を生成する問題データ
    {json.dumps(questions_for_prompt, ensure_ascii=False, indent=2)}
    """


# --- Streamlit アプリケーション ---

st.title("Azure AZ-104 模擬試験アプリ")

# セッション状態で問題と回答を管理
st.session_state.setdefault("questions", None)
st.session_state.setdefault("user_answers", {})
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("explanations", {})

# サイドバー
button_placeholder = st.sidebar.empty()

st.sidebar.header("出題カテゴリ")
with st.sidebar.expander("カテゴリを選択してください", expanded=True):
    # チェックボックスを一つずつ作成
    selected_cats = [category for category in AZ104_CATEGORIES if st.checkbox(category, key=f"cb_{category}")]

st.sidebar.header("モデル選択")
model_choice = st.sidebar.radio(
    "Gemini API model:",
    ("Gemini 2.5 Flash Lite", "Gemini 2.5 Flash", "Gemini 2.5 Pro"),
    key="model_choice",
)

if button_placeholder.button(f"新しい問題を{QUIZ_NUM}問生成", key="generate_button"):
    st.session_state.submitted = False
    with st.spinner("Geminiが問題を生成中です..."):
        prompt = create_prompt(selected_cats)
        try:
            response = client.models.generate_content(
                model=model_dict[model_choice],
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

# --- Streamlit アプリケーション (結果表示部分) ---

# st.session_state.setdefault("questions", None) の近くに以下を追加
st.session_state.setdefault("explanations", {})

# ... (問題生成までのコードは省略) ...

# 回答が送信されたら結果を表示
if st.session_state.submitted:
    # 1. まず全問の採点を行い、結果をリストに保存
    results_data = []
    correct_count = 0
    if "results_data" not in st.session_state:
        for i, q in enumerate(st.session_state.questions):
            user_ans = sorted(st.session_state.user_answers.get(i, []))
            correct_ans = sorted(q.answer)
            is_correct = user_ans == correct_ans
            if is_correct:
                correct_count += 1

            results_data.append(
                {
                    "is_correct": is_correct,
                    "user_ans": user_ans,
                    "correct_ans": correct_ans,
                },
            )
        st.session_state.results_data = results_data
        st.session_state.correct_count = correct_count
    else:
        # rerun後もデータを維持
        results_data = st.session_state.results_data
        correct_count = st.session_state.correct_count

    # 2. 正答率を一番上に表示
    st.header("結果発表")
    score_percent = correct_count / len(st.session_state.questions)
    st.header(f"正答率: {correct_count} / {len(st.session_state.questions)} ({score_percent:.0%})")

    # 3. 解説を生成したい問題を選択するUI
    with st.expander("解説を生成したい問題を選択してください", expanded=True), st.form("explain_form"):
        indices_to_select = [
            i for i in range(len(st.session_state.questions)) if i not in st.session_state.explanations
        ]

        if not indices_to_select:
            st.info("すべての問題の解説が生成済みです。")
        else:
            selected_indices = []
            st.write("解説を見たい問題にチェックを入れてください。（不正解の問題はデフォルトで選択されています）")
            for i in indices_to_select:
                q = st.session_state.questions[i]
                # 不正解の問題はデフォルトでチェック
                is_checked_by_default = not results_data[i]["is_correct"]
                if st.checkbox(f"問題 {i + 1}: {q.question[:50]}...", key=f"cb_exp_{i}", value=is_checked_by_default):
                    selected_indices.append(i)

        submitted = st.form_submit_button("選択した問題の解説をまとめて生成")

        if submitted and selected_indices:
            with st.spinner(f"{len(selected_indices)}件の解説をまとめて生成中です..."):
                batch_prompt = create_batch_explanation_prompt(
                    selected_indices,
                    st.session_state.questions,
                    results_data,
                )
                try:
                    response = client.models.generate_content(
                        model=model_dict[model_choice],
                        contents=batch_prompt,
                        config={
                            "response_mime_type": "application/json",
                            # ★★★ ここを新しいモデルに変更 ★★★
                            "response_schema": list[DetailedExplanationModel],
                        },
                    )
                    # 生成された解説オブジェクトをそのままsession_stateに保存
                    for expl_obj in response.parsed:
                        st.session_state.explanations[expl_obj.question_index] = expl_obj

                    st.success("解説の生成が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"解説の生成中にエラーが発生しました: {e}")

    # 4. 各問題の結果と、生成済みの解説を表示
    st.header("各問題の結果")
    for i, q in enumerate(st.session_state.questions):
        result = results_data[i]
        with st.container(border=True):
            st.subheader(f"問題 {i + 1} ({q.category})")
            st.write(q.question)

            if result["is_correct"]:
                st.success("正解！")
            else:
                st.error("不正解")

            st.write(f"**あなたの回答**: {', '.join(result['user_ans'])}")
            st.write(f"**正解**: {', '.join(result['correct_ans'])}")

            # 生成済みの解説があれば表示
            if i in st.session_state.explanations:
                with st.expander("詳細な解説を見る"):
                    explanation_data = st.session_state.explanations[i]

                    st.success(f"**要点:** {explanation_data.overall_summary}")

                    st.write("#### あなたの回答へのフィードバック")
                    st.warning(explanation_data.user_feedback)

                    st.write("#### 正解の解説")
                    st.write(explanation_data.correct_answer_explanation)

                    st.write("#### 他の選択肢について")
                    st.write(explanation_data.analysis_of_other_options)

    # for i, q in enumerate(st.session_state.questions):
    #     user_ans = sorted(st.session_state.user_answers.get(i, []))
    #     correct_ans = sorted(q.answer)
    #     is_correct = user_ans == correct_ans

    #     st.subheader(f"問題 {i + 1}")
    #     st.write(q.question)

    #     if is_correct:
    #         st.success("正解!")
    #         correct_count += 1
    #     else:
    #         st.error("不正解")

    #     st.write(f"あなたの回答: {', '.join(user_ans)}")
    #     st.write(f"正解: {', '.join(correct_ans)}")

    #     # 解説生成
    #     reason = st.session_state.get(f"reason_{i}", "")
    #     if not is_correct or reason:
    #         with st.spinner(f"問題{i + 1}の解説を生成中..."):
    #             options_for_prompt = {opt.id: opt.text for opt in q.options}

    # explanation_prompt = f"""
    # 以下のAzureの問題について、なぜこれが正解なのか解説してください。
    # ユーザーの回答や記入内容も踏まえて、特に間違っている点を重点的に説明してください。

    # ## 問題
    # {q.question}

    # ## 選択肢
    # {json.dumps(options_for_prompt, ensure_ascii=False)}

    # ## 正解
    # {", ".join(correct_ans)}

    # ## ユーザーの回答
    # {", ".join(user_ans)}

    # ## ユーザーが記入した理由
    # {reason if reason else "記載なし"}
    # """
    #             try:
    #                 explanation_response = client.models.generate_content(
    #                     model=model_dict[model_choice],
    #                     contents=explanation_prompt,
    #                 )
    #                 st.info(f"解説:\n{explanation_response.text}")
    #             except Exception as e:
    #                 st.warning(f"解説の生成に失敗しました: {e}")

    # st.header(f"正答率: {correct_count / len(st.session_state.questions):.0%}")
