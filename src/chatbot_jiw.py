"""주제 ③ — JIW 34 경보 설명 챗봇: 근거 카드와 실험 결과 요약(facts)만으로 답하는 대화형 인터페이스.

설계 원칙 (docs/chatbot_design_JIW.md)
- 챗봇은 **새로운 판단을 하지 않는다.** 경보 판정은 28번 규칙(코드)이 하고, 챗봇은 그 근거(32번 근거 카드)와 결과 요약(facts)을 풀어 설명한다.
- 모든 답은 facts 번호([F3] 같은 인용)를 단다. facts에 없는 질문은 "제공된 근거에 없습니다"로 답한다.
- 원인(모터인지 펌프인지, 어떤 부품인지)을 단정하지 않는다. 점검 권고는 [가정]이다.
- 답에 들어 있는 숫자는 facts·선택한 카드에 있는 값이어야 한다(`verify_numbers`). 없으면 경고를 붙인다.
- 백엔드 두 가지: `OfflineBackend`(키워드 라우팅 + 템플릿, 외부 호출 없음, 결정적, 기본)와 `LLMBackend`(외부 LLM API, 선택).
  LLM 백엔드는 facts 요약 텍스트와 질문·대화 이력만 보낸다. 원 데이터·윈도우 점수는 보내지 않는다.
  외부 API 사용 가능 여부는 대회 규정 확인이 먼저다. 기본은 오프라인이다.
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

import paths

REFUSE = "제공된 근거에 없습니다."
RES28, RES32, RES33 = (paths.RESULTS / n for n in ("28_model_ensemble_JIW", "32_alarm_explain_JIW", "33_early_warning_JIW"))
SPLIT_KO = {"group_kfold_seg": "group_kfold", "time_block": "time_block"}


# ------------------------------------------------------------------ facts
def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def build_facts() -> dict[str, str]:
    """results/28·32·33 CSV에서 읽은 수치와 고정 문구로 facts를 만든다 {번호: 문장}."""
    cv = pd.read_csv(RES28 / "cv_scores.csv")
    g = cv.groupby(["model", "split"]).mean(numeric_only=True)
    pair = lambda m, col, f=lambda v: f"{v:.3g}": " / ".join(f(g.loc[(m, s), col]) for s in ("group_kfold_seg", "time_block"))
    hour = lambda m: " / ".join(f"{g.loc[(m, s), 'fpr_segment'] * 452:.1f}" for s in ("group_kfold_seg", "time_block"))
    lead = pd.read_csv(RES33 / "summary.csv", index_col=[0, 1])
    fa = pd.read_csv(RES33 / "false_alerts.csv", index_col=0)
    eta = pd.read_csv(RES33 / "eta_errors.csv")
    att = pd.read_csv(RES32 / "attribution.csv")
    cards = pd.read_csv(RES32 / "cards.csv")
    real = cards[cards["label"] == 1]
    top = real["top_channel"].value_counts(normalize=True)
    ko = {"AI0_Vibration": "상부 진동", "AI1_Vibration": "하부 진동", "AI2_Current": "전류"}
    lead_txt = lambda rule: " / ".join(f"{lead.loc[(rule, d), '선행 시간 중앙값(분)']:.2f}" for d in (2.0, 5.0, 10.0))
    sev = lambda m, c: pair(m, c, _pct)
    ret2 = " / ".join(f"{g.loc[('cnn&mcd', s), 'inj_sev2'] / g.loc[('cnn', s), 'inj_sev2'] * 100:.0f}%" for s in ("group_kfold_seg", "time_block"))
    ret8 = " / ".join(f"{g.loc[('cnn&mcd', s), 'inj_sev8'] / g.loc[('cnn', s), 'inj_sev8'] * 100:.0f}%" for s in ("group_kfold_seg", "time_block"))
    fp_hi = pair("cnn", "fpr_high_load", _pct)
    fp_lo = pair("cnn", "fpr_low_load", _pct)
    state_hi = pair("cnn [상태별]", "fpr_high_load", _pct)
    cnn_hit = att[att["sev"] >= 2].groupby(["kind", "sev"])["cnn_hit"].mean()   # 유형 × 강도 칸별 적중률
    eta_med = eta.groupby("dur_min")[["true_remaining_min", "abs_err_min"]].median()
    gain = (lead.loc[("추세 λ0.4 (주 규칙)",), "선행 시간 중앙값(분)"] - lead.loc[("CNN 단독(주의)",), "선행 시간 중앙값(분)"])
    return {
        "F1": "경보 규칙: 주의 = CNN 점수가 임계를 넘음, 경보 = CNN과 MCD 점수가 모두 임계를 넘음. 경보는 항상 주의의 부분집합이다. "
              "CNN은 신호의 다음 값을 예측해 틀린 정도를 보고, MCD는 윈도우 통계 18개가 정상 무리에서 멀리 떨어진 정도를 본다.",
        "F2": "임계는 학습용 정상 점수의 상위 1%(q99)로만 정하고 이상 데이터로 조정하지 않는다. 점수는 임계로 나눈 배수로 표시한다(1배가 임계).",
        "F3": f"CNN 단독(주의 단계) 세그먼트 오경보율은 {pair('cnn', 'fpr_segment', _pct)}(group_kfold / time_block), 시간당 약 {hour('cnn')}건이다. "
              f"실제 이상 세그먼트는 전부 탐지했다.",
        "F4": f"CNN과 MCD를 모두 넘어야 하는 경보 단계의 세그먼트 오경보율은 {pair('cnn&mcd', 'fpr_segment', _pct)}, 시간당 약 {hour('cnn&mcd')}건이다. "
              f"실제 이상 세그먼트는 전부 탐지했다(미탐 0건).",
        "F5": f"MCD 단독 세그먼트 오경보율은 {pair('mcd', 'fpr_segment', _pct)}, CNN 또는 MCD(OR)는 {pair('cnn|mcd', 'fpr_segment', _pct)}로 오경보가 늘어 채택하지 않았다.",
        "F6": f"약한 이상 탐지: 합성 강도 2에서 경보(AND)의 탐지율은 CNN 단독 대비 {ret2}(group_kfold / time_block)만 유지되고, 강도 8에서는 {ret8}다. "
              f"합성 강도 1 이하의 약한 이상은 경보도 주의도 거의 잡지 못한다(CNN 단독 강도 1 탐지율 {sev('cnn', 'inj_sev1')}, 경보 {sev('cnn&mcd', 'inj_sev1')}).",
        "F7": f"오경보는 고부하 운전 구간에 몰린다. CNN 단독 고부하 오경보 {fp_hi}, 저부하 {fp_lo}(group_kfold / time_block). "
              f"운전 상태별 임계로 바꿔도 고부하 오경보는 {state_hi}로 줄지 않아 채택하지 않았다. 경보 단계(CNN AND MCD)가 이 오경보를 지운다.",
        "F8": f"조기경보(합성 열화 시나리오): 열화가 완성되기 전에 알린 선행 시간 중앙값은 주의 {lead_txt('CNN 단독(주의)')}분, 경보 {lead_txt('경보(CNN AND MCD)')}분이다(열화 2 / 5 / 10분 기준). "
              f"선행 시간은 열화 기간의 약 70%라서 실제 설비의 열화 속도를 모르면 '몇 분 전'을 말할 수 없다.",
        "F9": f"정상 구간에서 거짓 사건은 주의 시간당 {fa.loc['CNN 단독(주의)', '시간당 사건']:.1f}건, 경보 시간당 {fa.loc['경보(CNN AND MCD)', '시간당 사건']:.1f}건이다"
              f"(정상 약 {fa.loc['CNN 단독(주의)', '관측 시간(h)']:.1f}시간 분량, 사건 수가 적어 불확실하다).",
        "F10": f"위험 상승 추세 규칙은 CNN 단독보다 선행 시간이 중앙값 기준 {gain.iloc[0]:.2f} / {gain.iloc[1]:.2f} / {gain.iloc[2]:.2f}분 길 뿐이라 선택 사항이다.",
        "F11": f"'약 N분 후 경보 도달' 추정은 쓰지 않는다. 도달 시간 추정 오차(중앙값 {eta_med['abs_err_min'].min():.1f}~{eta_med['abs_err_min'].max():.1f}분)가 "
               f"실제 남은 시간(중앙값 {eta_med['true_remaining_min'].min():.1f}~{eta_med['true_remaining_min'].max():.1f}분)과 같거나 크다.",
        "F12": f"근거 카드 정합성: 합성 이상에서 CNN 예측 오차 비중 1위 채널이 주입한 채널과 일치한 비율은 강도 2 이상의 유형·강도별 칸에서 {cnn_hit.min() * 100:.0f}~{cnn_hit.max() * 100:.0f}%다(합성 검증이며 실제 원인 진단 성능이 아니다).",
        "F13": f"실제 이상 창 {len(real)}개의 오차 비중 1위 채널은 " + ", ".join(f"{ko[k]} {v * 100:.0f}%" for k, v in top.items()) +
               "이고 세그먼트마다 다르다. 이상 하나가 여러 신호에 걸쳐 나타나므로 카드는 '어느 신호가 벗어났는지'까지만 말한다.",
        "F14": "점검 권고(가정): 상부 진동 우세 → 상단 체결부·커플링·정렬, 하부 진동 우세 → 하단 베이스·마운트, 전류 우세 → 전원·부하 변동·토출 압력, 진동과 전류 동시 → 구동부 전반 종합 점검. "
               "가이드북과 일반 설비 상식에서 온 가정이며 이 데이터로 검증하지 않았다.",
        "F15": "센서 부착 부위(모터 몸체인지 펌프 몸체인지)를 데이터로 확정할 수 없어 원인이 모터인지 펌프인지, 어떤 부품인지는 구분할 수 없다. 탐지 대상은 '유압펌프 모터-펌프 구동부 이상'으로 쓴다.",
        "F16": "한계: 실제 이상 이벤트가 1건(정상 하루 77분)이라 결과는 이 이벤트의 기술이다. 정상에서 이상으로 넘어가는 구간이 없어 '이상 몇 분 전 경보'는 실데이터로 검증할 수 없다. "
               "합성 열화·합성 이상은 방법 작동 확인이며 실제 고장 예측 성능이 아니다.",
        "F17": "조용한 이상(세그먼트 3·19·20)은 진동 전용 입력·규칙은 놓치지만 전류 채널을 입력에 넣은 모델은 모두 탐지한다. 현장 센서 구성에서 전류는 필수다.",
    }


def facts_text(facts: dict[str, str]) -> str:
    return "\n".join(f"[{k}] {v}" for k, v in facts.items())


def load_cards() -> pd.DataFrame:
    return pd.read_csv(RES32 / "cards.csv")


# ------------------------------------------------------------------ 숫자 검증
_NUM = re.compile(r"\d+(?:\.\d+)?")


def verify_numbers(answer: str, allowed_text: str) -> list[str]:
    """답에 있고 facts·카드 텍스트에 없는 숫자 목록. 인용 [F3]·목록 번호는 제외한다."""
    body = re.sub(r"\[F\d+\]", "", answer)
    body = re.sub(r"(?m)^\s*\d+[.)]\s", "", body)
    allowed = set(_NUM.findall(allowed_text))
    return sorted({n for n in _NUM.findall(body) if n not in allowed})


UNSAFE = [   # 단정 표현: 원인 단정, 시점 예측 (규칙 4·5 위반)
    re.compile(r"(모터|펌프|베어링|축|커플링)[^.\n]{0,6}(고장|마모|결함|이상)(입니다|이다|이에요|때문|으로 확인)"),
    re.compile(r"\d+(?:\.\d+)?\s*분\s*(후|뒤)[^.\n]{0,8}(경보|이상|고장|도달)"),
]


def unsafe_claims(answer: str) -> list[str]:
    """원인 단정·시점 예측으로 보이는 문장 조각 (정규식 휴리스틱이라 놓칠 수 있다)"""
    return [m.group(0) for r in UNSAFE for m in r.finditer(answer)]


# ------------------------------------------------------------------ 백엔드
INTENTS = [   # (이름, 키워드, 사용할 facts)
    ("cause_pin", ("원인", "베어링", "고장", "모터가", "펌프가", "부품", "어디가"), ("F15", "F12", "F13")),
    ("eta", ("몇 분", "언제 경보", "얼마 후", "도달", "예측해"), ("F11", "F8", "F16")),
    ("early", ("조기", "선행", "미리", "예지", "열화"), ("F8", "F9", "F10", "F16")),
    ("false_alarm", ("오경보", "거짓", "얼마나 자주", "자주", "고부하", "저부하"), ("F3", "F4", "F7", "F9")),
    ("missed", ("미탐", "놓치", "못 잡", "조용한", "약한"), ("F6", "F17", "F16")),
    ("action", ("조치", "점검", "어떻게 해", "뭘 해", "무엇을 해"), ("F14", "F15")),
    ("rule", ("규칙", "기준", "임계", "단계", "주의", "경보가", "어떻게 판정"), ("F1", "F2", "F3", "F4")),
    ("compare", ("or", "단독", "비교", "mcd", "cnn"), ("F3", "F4", "F5")),
    ("limit", ("한계", "신뢰", "정확", "믿을", "검증"), ("F16", "F12", "F15")),
]


class OfflineBackend:
    """키워드 라우팅 + 템플릿. facts 문장을 그대로 인용하므로 숫자가 facts와 어긋날 수 없다."""
    name = "offline"

    def answer(self, question: str, history: list[dict], facts: dict[str, str], card: dict | None) -> str:
        q = question.lower()
        if card and any(k in q for k in ("이 경보", "왜 울", "왜 경보", "이유", "이 카드", "지금 경보")):
            return (f"선택한 경보의 근거 카드입니다.\n{card['text']}\n"
                    f"예측 오차가 몰린 신호는 '{card['where']}'이지만 이것이 원인 부위라는 뜻은 아닙니다 [F13][F15]. 점검 권고는 가정입니다 [F14].")
        for name, keys, ids in INTENTS:
            if any(k in q for k in keys):
                lines = [f"- {facts[i]} [{i}]" for i in ids]
                head = {"cause_pin": "원인(모터/펌프/부품)은 이 데이터로 구분할 수 없습니다. 근거로 말할 수 있는 것은 아래입니다.",
                        "eta": "'몇 분 후'를 말할 수 없습니다. 아래가 이유입니다."}.get(name, "근거 요약입니다.")
                return head + "\n" + "\n".join(lines)
        return f"{REFUSE} 다룰 수 있는 주제: 경보 규칙, 오경보, 미탐, 조기경보, 점검 권고, 한계. 선택한 경보가 있으면 '왜 울렸어?'라고 물어보세요."


SYSTEM_PROMPT = """당신은 프레스 유압펌프 이상탐지 결과를 설명하는 챗봇입니다. 아래 [FACTS]와 [CARD]만 근거로 한국어로 답합니다.
규칙:
1. 답에 쓰는 모든 문장은 [FACTS]나 [CARD]에서 온 것이어야 하고, 사실에는 [F3] 같은 번호를 붙입니다.
2. [FACTS]에 없는 질문은 "제공된 근거에 없습니다."라고 답합니다. 추측하지 않습니다.
3. 숫자는 [FACTS]·[CARD]에 있는 값만 그대로 씁니다. 새 숫자를 계산하거나 만들지 않습니다.
4. 이상의 원인이 모터인지 펌프인지, 어떤 부품인지 단정하지 않습니다. 점검 권고는 항상 '가정'이라고 밝힙니다.
5. "몇 분 후 이상이 생긴다" 같은 시점 예측은 하지 않습니다. 근거(F11)를 설명합니다.
6. 짧고 구체적으로 답합니다."""


class LLMBackend:
    """외부 LLM API 백엔드(선택). 환경 변수 PDM_LLM_MODEL(기본값은 아래 코드), 인증은 SDK 기본 경로.
    facts 요약 텍스트와 대화 이력만 보낸다. 이 경로는 키가 필요해 노트북 실행에서는 시험하지 않았다(34번 §4)."""
    name = "llm"

    def __init__(self, client=None, model: str | None = None):
        if client is None:
            import anthropic   # 선택 의존성: pip install anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("PDM_LLM_MODEL", "claude-opus-5-5")

    def answer(self, question: str, history: list[dict], facts: dict[str, str], card: dict | None) -> str:
        card_txt = card["text"] if card else "(선택한 경보 없음)"
        system = f"{SYSTEM_PROMPT}\n\n[FACTS]\n{facts_text(facts)}\n\n[CARD]\n{card_txt}"
        messages = history + [{"role": "user", "content": question}]
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=4000, system=system, messages=messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return f"모델이 이 질문에 답하지 않았습니다. {REFUSE}"
        return next((b.text for b in resp.content if b.type == "text"), REFUSE)


# ------------------------------------------------------------------ 세션
class ChatSession:
    def __init__(self, backend=None, facts: dict[str, str] | None = None, cards: pd.DataFrame | None = None):
        self.backend = backend or OfflineBackend()
        self.facts = facts or build_facts()
        self.cards = cards if cards is not None else load_cards()
        self.card: dict | None = None
        self.history: list[dict] = []

    def select_card(self, i: int):
        """실제 이상 창·오경보 창 카드 중 i번째를 설명 대상으로 고른다"""
        self.card = self.cards.iloc[i].to_dict()
        return self.card

    def ask(self, question: str) -> dict:
        allowed = facts_text(self.facts) + "\n" + (self.card["text"] if self.card else "")
        try:
            ans = self.backend.answer(question, self.history, self.facts, self.card)
        except Exception as e:   # 외부 호출 실패 시 오프라인으로 대체(오류 종류만 알린다)
            ans = f"({self.backend.name} 백엔드 오류 {type(e).__name__}, 오프라인 답으로 대체)\n" + OfflineBackend().answer(question, self.history, self.facts, self.card)
        bad = verify_numbers(ans, allowed)
        unsafe = unsafe_claims(ans)
        cited = sorted(set(re.findall(r"\[(F\d+)\]", ans)), key=lambda s: int(s[1:]))
        if bad:
            ans += f"\n⚠ 근거에서 확인되지 않는 숫자: {', '.join(bad)}"
        if unsafe:
            ans += f"\n⚠ 원인·시점을 단정하는 표현이 있습니다: {', '.join(unsafe)}"
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": ans}]
        return {"question": question, "answer": ans, "cited": cited, "unsupported_numbers": bad, "unsafe_claims": unsafe,
                "refused": ans.startswith(REFUSE) or REFUSE in ans.splitlines()[0]}


# ------------------------------------------------------------------ 평가 질문
EVAL_QUESTIONS = [   # (질문, 종류, 반드시 인용할 facts)
    ("경보랑 주의는 뭐가 달라?", "answer", ("F1",)),
    ("임계값은 어떻게 정해?", "answer", ("F2",)),
    ("오경보는 얼마나 자주 나?", "answer", ("F3", "F4")),
    ("고부하에서 오경보가 왜 많아?", "answer", ("F7",)),
    ("약한 이상도 잡아?", "answer", ("F6",)),
    ("조기경보는 몇 분 전에 울려?", "answer", ("F8",)),
    ("미탐은 어떤 경우야?", "answer", ("F6", "F17")),
    ("어떻게 점검해야 해?", "answer", ("F14",)),
    ("이 결과를 얼마나 믿을 수 있어?", "answer", ("F16",)),
    ("CNN 단독이랑 MCD 단독 비교해줘", "answer", ("F3", "F5")),
    ("몇 분 후에 경보가 울릴지 예측해줘", "decline", ("F11",)),
    ("고장 원인이 모터 베어링이야?", "decline", ("F15",)),
    ("내일 날씨 어때?", "refuse", ()),
    ("주식 추천해줘", "refuse", ()),
]


def evaluate(backend, facts=None, cards=None) -> pd.DataFrame:
    """질문 세트에서 인용·거절·숫자 근거를 점검한다. 종류: answer(인용 필요), decline(단정 거절 + 근거 설명), refuse(범위 밖은 근거 없음)"""
    rows = []
    for q, kind, need in EVAL_QUESTIONS:
        r = ChatSession(backend, facts, cards).ask(q)
        ok_cite = all(n in r["cited"] for n in need) if need else True
        ok = ok_cite if kind != "refuse" else r["refused"]
        if kind == "decline":
            ok = ok_cite and any(k in r["answer"] for k in ("구분할 수 없", "말할 수 없", "쓰지 않"))
        rows.append({"질문": q, "종류": kind, "인용": ",".join(r["cited"]), "필수 인용 충족": ok_cite, "거절": r["refused"],
                     "근거 없는 숫자": len(r["unsupported_numbers"]), "단정 표현": len(r["unsafe_claims"]),
                     "통과": ok and not r["unsupported_numbers"] and not r["unsafe_claims"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":   # python src/chatbot_jiw.py [--llm]
    sess = ChatSession(LLMBackend() if "--llm" in sys.argv else OfflineBackend())
    print(f"백엔드: {sess.backend.name}. '/card N'으로 경보 카드 선택(0~{len(sess.cards) - 1}), 빈 줄이면 종료.")
    while (q := input("질문> ").strip()):
        if q.startswith("/card"):
            c = sess.select_card(int(q.split()[1]))
            print(c["text"])
            continue
        print(sess.ask(q)["answer"])
