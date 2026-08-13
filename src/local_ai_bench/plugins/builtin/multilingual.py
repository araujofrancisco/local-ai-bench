"""Multilingual benchmark — the model must understand and answer in-language.

Each case asks a question in a non-English language and grades two things:

1. **Comprehension** — the reply surfaces facts from the expected answer
   (in-language keyword recall).
2. **Language fidelity** — the reply stays in the prompt's language rather than
   drifting to English. For scripts that Unicode ranges can identify decisively
   (CJK, Hangul, Arabic, Devanagari, Cyrillic, Greek), this is a hard check;
   for Latin-script languages (es/fr/de/pt/it/nl) it checks that at least some
   non-ASCII Latin characters or stop-words appear, so a reply is penalised
   only when it has clearly switched languages.

Scoring is fully deterministic: Unicode range analysis plus keyword recall, with
no judge model required. This makes the benchmark cheap to run on local models.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import keyword_recall

# (name, regex) — Unicode block ranges keyed by language code.
_SCRIPTS: dict[str, list[str]] = {
    "ja": [r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"],
    "zh": [r"[\u3400-\u4dbf\u4e00-\u9fff]"],
    "ko": [r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"],
    "ar": [r"[\u0600-\u06ff\u0750-\u077f]"],
    "hi": [r"[\u0900-\u097f]"],
    "ru": [r"[\u0400-\u04ff]"],
    "el": [r"[\u0370-\u03ff]"],
    # Latin-script languages share ranges; stop-words below disambiguate.
    "es": [r"[áéíóúüñ¿¡]"],
    "fr": [r"[àâçéèêëîïôùûüœÿ]"],
    "de": [r"[äöüß]"],
    "it": [r"[àèéìòóù]"],
    "pt": [r"[ãõáéíóúâêôàç]"],
    "nl": [r"[ëïöüáéíóú]"],
}

# Accented/full-width markers that only appear when the model stays in the
# prompt's script (a pure-English reply for es/fr/etc. has none of these).
_LATIN_SCRIPT_STABILITY: dict[str, list[str]] = {
    "es": ["á", "é", "í", "ó", "ú", "ñ", "¿", "¡"],
    "fr": ["à", "ç", "é", "è", "ê", "î", "ô", "û", "ï"],
    "de": ["ä", "ö", "ü", "ß"],
    "it": ["à", "è", "é", "ì", "ò", "ù"],
    "pt": ["ã", "õ", "á", "é", "í", "ó", "ú", "â", "ê", "ô"],
    "nl": ["ë", "ï", "é", "é"],
}

_CASES = [
    {
        "id": "ml_ja_0001",
        "language": "ja",
        "prompt": "日本の首都はどこですか？",
        "keywords": ["東京"],
    },
    {
        "id": "ml_zh_0002",
        "language": "zh",
        "prompt": "中国的首都是哪里？",
        "keywords": ["北京"],
    },
    {
        "id": "ml_ar_0003",
        "language": "ar",
        "prompt": "ما هي عاصمة فرنسا؟",
        "keywords": ["باريس"],
    },
    {
        "id": "ml_es_0004",
        "language": "es",
        "prompt": "¿Cuál es la capital de España?",
        "keywords": ["madrid"],
    },
    {
        "id": "ml_fr_0005",
        "language": "fr",
        "prompt": "Quelle est la capitale de l'Italie ?",
        "keywords": ["rome", "italie"],
    },
    {
        "id": "ml_de_0006",
        "language": "de",
        "prompt": "Was ist die Hauptstadt von Deutschland?",
        "keywords": ["berlin"],
    },
    {
        "id": "ml_ru_0007",
        "language": "ru",
        "prompt": "Сколько ног у паука?",
        "keywords": ["восемь"],
    },
    {
        "id": "ml_ko_0008",
        "language": "ko",
        "prompt": "한국의 수도는 어디인가요?",
        "keywords": ["서울"],
    },
]


def _has_script(text: str, language: str) -> bool:
    patterns = _SCRIPTS.get(language, [])
    return any(re.search(p, text) for p in patterns)


def _latin_stable(text: str, language: str) -> bool:
    markers = _LATIN_SCRIPT_STABILITY.get(language)
    return markers is None or any(m in text for m in markers)


def _language_kept(prompt: str, response: str, language: str) -> bool:
    """True when the reply plausibly stays in the prompt's language."""
    if language in _SCRIPTS and language not in _LATIN_SCRIPT_STABILITY:
        # Non-Latin script: the response must use it at all.
        return _has_script(response, language)
    # Latin-script languages: require an accent/diacritic or a strong stop-word.
    return _latin_stable(response, language)


class MultilingualPlugin(BaseTextPlugin):
    id: ClassVar[str] = "multilingual"
    name: ClassVar[str] = "Multilingual"
    description: ClassVar[str] = (
        "Model must answer in-language across several languages."
    )
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.MULTILINGUAL
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model) -> bool:  # noqa: ANN001
        return True

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={
                    "prompt": spec["prompt"],
                    "language": spec["language"],
                },
                expected={
                    "keywords": spec["keywords"],
                    "language": spec["language"],
                },
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question in the same language they "
                        "wrote it in. Be concise."
                    ),
                },
                {"role": "user", "content": case.input["prompt"]},
            ],
            "options": {"temperature": 0.0, "num_predict": 64},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        language = case.input["language"]
        keywords = case.expected["keywords"]
        recall = keyword_recall(response.text, keywords)
        language_kept = _language_kept(case.input["prompt"], response.text, language)

        # Drifting to another language caps comprehension credit at 50%.
        score = recall if language_kept else round(recall * 0.5, 4)

        return Evaluation(
            score=score,
            passed=score == 1.0,
            metrics={
                "language": language,
                "language_kept": language_kept,
                "keyword_recall": recall,
            },
        )