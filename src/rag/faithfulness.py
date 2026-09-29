"""Mesure de fidélité (faithfulness) des réponses générées en Phase 3 (Phase 4).

La garde-fou de generate.py décide seulement SI le système doit répondre. Ici, pour les réponses où il
a choisi de répondre, on vérifie qu'il n'a rien ajouté qui ne soit pas dans les passages cités : un
second appel LLM (juge) découpe la réponse en affirmations factuelles et juge chacune indépendamment,
plutôt que de comparer des chaînes de caractères — une affirmation peut être vraie sans être un extrait
mot pour mot du contexte (paraphrase, reformulation).
"""
from rag.generate import LLMClient, build_context, extract_json_object

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You are given CONTEXT passages and an \
ANSWER that is supposed to be grounded only in that context. Break the answer down into its individual \
factual claims, and for each one decide whether it is directly stated or clearly implied by the context.

Rules:
- A claim is "supported" only if the context actually contains that information (paraphrasing is fine,
  adding information the context does not have is not).
- Ignore filler sentences with no factual content (e.g. "Here is how to do it:") — do not list them as claims.
- If the answer has no factual claims at all (e.g. it just says "I don't know"), respond with an empty list.

Respond with JSON only, no markdown fences, no text before or after:
{"claims": [{"claim": "<claim text>", "supported": true or false}, ...]}"""


def _parse_judge_response(raw: str) -> dict:
    data = extract_json_object(raw)
    if "claims" not in data or not isinstance(data["claims"], list):
        raise ValueError(f"Réponse du juge sans liste 'claims' : {raw!r}")
    for claim in data["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("supported"), bool):
            raise ValueError(f"Affirmation sans clé 'supported' booléenne : {claim!r}")
    return data


def judge_faithfulness(client: LLMClient, answer: str, passages: list[tuple[str, str]]) -> dict:
    """passages : (clé, texte) des passages CITÉS par la réponse (pas tout le contexte récupéré) — on ne
    juge la fidélité que par rapport à ce que le modèle a dit s'être basé dessus.

    Renvoie {"score": float dans [0, 1], "claims": [{"claim": str, "supported": bool}, ...]}.
    `score` = part des affirmations soutenues ; 1.0 si la réponse n'a aucune affirmation factuelle
    détectée (ex. une réponse vide ou purement conversationnelle).
    """
    labels = [f"S{i + 1}" for i in range(len(passages))]
    context = build_context(list(zip(labels, (text for _, text in passages))))
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    parsed = _parse_judge_response(client.complete(JUDGE_SYSTEM_PROMPT, user))
    claims = parsed["claims"]
    score = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0
    return {"score": score, "claims": claims}
