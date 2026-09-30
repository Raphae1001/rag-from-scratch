"""Fine-tuning LoRA contrastif du modèle d'embeddings (Phase 6).

Adapte `all-MiniLM-L6-v2` au corpus avec une loss contrastive (`MultipleNegativesRankingLoss` : chaque
paire (requête, document positif) du batch sert de négatif implicite aux autres paires du même batch —
pas besoin d'annoter des négatifs explicites). Seuls de petits adaptateurs LoRA sont entraînés (rang 16
sur les projections query/value de l'attention), le modèle de base reste gelé : rapide, peu de mémoire,
et le résultat reste un modèle `sentence-transformers` standard rechargeable.

Usage :
    python scripts/make_training_pairs.py           # génère data/train/pairs.jsonl (une fois)
    python scripts/finetune_embeddings.py            # -> models/finetuned-minilm-lora/
"""
import argparse
import json
from pathlib import Path

from _common import ROOT

PAIRS = ROOT / "data" / "train" / "pairs.jsonl"


def load_pairs():
    from datasets import Dataset

    pairs = [json.loads(line) for line in PAIRS.open(encoding="utf-8")]
    return Dataset.from_dict({"anchor": [p["query"] for p in pairs], "positive": [p["text"] for p in pairs]})


def plot_loss(log_history: list[dict], out_path):
    import matplotlib
    matplotlib.use("Agg")  # pas d'affichage interactif : on écrit directement un fichier
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (MultipleNegativesRankingLoss)")
    ax.set_title("Fine-tuning LoRA — courbe de loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output", default=str(ROOT / "models" / "finetuned-minilm-lora"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    if not PAIRS.exists():
        raise SystemExit(f"{PAIRS} manquant : lancer scripts/make_training_pairs.py d'abord")

    import torch
    from peft import LoraConfig, TaskType
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device} (voir README pour le temps mesuré sur CPU)")

    model = SentenceTransformer(args.base_model, device=device)
    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_r, lora_alpha=args.lora_r * 2,
                              lora_dropout=0.1, target_modules=["query", "value"])
    model.add_adapter(peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"paramètres entraînables : {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_dataset = load_pairs()
    loss_fn = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / "results" / "phase6_training_run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=train_dataset, loss=loss_fn)

    import time
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    plot_loss(trainer.state.log_history, results_dir / "phase6_loss_curve.png")
    (results_dir / "phase6_training_log.json").write_text(json.dumps({
        "device": device, "n_pairs": len(train_dataset), "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "lora_r": args.lora_r, "trainable_params": trainable, "total_params": total,
        "train_seconds": train_s, "log_history": trainer.state.log_history,
    }, indent=2, default=float), encoding="utf-8")

    print(f"entraînement terminé en {train_s:.1f}s sur {device} ({len(train_dataset)} paires, {args.epochs} époques)")
    print(f"modèle sauvegardé -> {out}")
    print(f"courbe de loss -> {results_dir / 'phase6_loss_curve.png'}")


if __name__ == "__main__":
    main()
