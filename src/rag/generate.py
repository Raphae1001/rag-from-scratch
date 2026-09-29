"""Génération de réponses sourcées avec garde-fou anti-hallucination (Phase 3).

Le modèle reçoit les passages retenus après reranking, étiquetés [S1], [S2], ... et doit répondre en
JSON strict : soit {"answerable": false} si le contexte ne permet pas de répondre, soit
{"answerable": true, "answer": "...", "sources": ["S1", "S3"]}. Passer par un format structuré plutôt que
de chercher "je ne sais pas" dans du texte libre rend la détection déterministe et testable sans appel
réseau (voir FakeLLMClient).
"""
import json
from typing import Protocol

SYSTEM_PROMPT = """You are a documentation search assistant. The "Question" is a query typed by a developer \
into a documentation search box — it may be a short keyword phrase (e.g. "field validator") rather than a \
full sentence. Treat it as a request for information about that topic, never as an ambiguous question that \
needs clarification: you must never ask the user a follow-up question, and you must always respond in the \
JSON format below, with no exceptions.

You answer ONLY using the numbered context passages provided (labeled [S1], [S2], ...). Never use outside
knowledge.

Rules:
- If the passages do not contain enough information about the topic, respond with exactly:
  {"answerable": false}
- Otherwise, respond with: {"answerable": true, "answer": "<your answer>", "sources": ["S1", "S3"]}
  where "sources" lists the labels of every passage you actually used to build the answer.
- The answer must be grounded only in the cited passages: do not add facts that are not in them.
- Respond with JSON only. No markdown fences, no text before or after the JSON object. Never ask a
  clarifying question instead of responding — if the topic is too broad to summarize, still set
  "answerable": true and give the best grounded overview the passages support."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Client réel (API Claude). Lit ANTHROPIC_API_KEY dans l'environnement si `api_key` n'est pas fourni."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None, max_tokens: int = 1024):
        import anthropic  # import tardif : évite la dépendance dure pour qui ne fait que du retrieval

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class FakeLLMClient:
    """Client DÉTERMINISTE pour les tests : renvoie une réponse scriptée selon un fragment présent dans
    le prompt utilisateur, sans appel réseau. `calls` garde l'historique pour les assertions."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        for fragment, response in self.responses.items():
            if fragment in user:
                return response
        raise KeyError(f"Aucune réponse scriptée ne correspond à ce prompt : {user[:200]!r}")


def build_context(passages: list[tuple[str, str]]) -> str:
    """passages : (label, texte), ex. [("S1", "..."), ("S2", "...")]."""
    return "\n\n".join(f"[{label}] {text}" for label, text in passages)


def extract_json_object(raw: str) -> dict:
    """Parse le premier objet JSON en tête de la réponse, en ignorant tout ce qui suit.

    En pratique, malgré une consigne « JSON only, no text before or after », le modèle ajoute parfois une
    explication après l'objet JSON (observé avec Haiku sur le set de non-réponse de la Phase 3).
    `json.loads` rejette toute la chaîne dans ce cas ; `raw_decode` ne lit que le premier objet JSON
    valide et laisse le reste, ce qui rend le parsing robuste à ce type d'écart mineur. Réutilisée par
    `faithfulness.py` (autre appel LLM à sortie JSON, même écart possible).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse du modèle non-JSON : {raw!r}") from e
    return data


def _parse_response(raw: str) -> dict:
    data = extract_json_object(raw)
    if "answerable" not in data:
        raise ValueError(f"Réponse du modèle sans clé 'answerable' : {raw!r}")
    if not isinstance(data["answerable"], bool):
        # ex. {"answerable": "false"} (string) : `not "false"` vaut False en Python (string non vide = truthy),
        # donc sans cette garde le code prendrait la branche "répondable" alors que le modèle voulait refuser.
        raise ValueError(f"'answerable' n'est pas un booléen JSON : {raw!r}")
    return data


def generate_answer(client: LLMClient, query: str, passages: list[tuple[str, str]]) -> dict:
    """passages : (clé de citation "path#section", texte), déjà triés par pertinence (post-reranking).

    Renvoie {"answerable": bool, "answer": str | None, "citations": list[str]} — les citations sont les
    vraies clés `path#section` (le mapping [S1]->clé se fait ici, le modèle ne voit jamais les clés
    réelles pour éviter qu'il en invente une qui ressemble à un vrai chemin).
    """
    labels = [f"S{i + 1}" for i in range(len(passages))]
    label_to_key = dict(zip(labels, (key for key, _ in passages)))
    context = build_context(list(zip(labels, (text for _, text in passages))))
    user = f"Context:\n{context}\n\nQuestion: {query}"

    parsed = _parse_response(client.complete(SYSTEM_PROMPT, user))  # garantit que parsed["answerable"] est un bool
    if not parsed["answerable"]:
        return {"answerable": False, "answer": None, "citations": []}

    citations = [label_to_key[s] for s in parsed.get("sources", []) if s in label_to_key]
    return {"answerable": True, "answer": parsed.get("answer"), "citations": citations}
