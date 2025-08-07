import argparse
import time
import os
import pandas as pd
import shutil

import matplotlib.pyplot as plt

from src.data import DataLoader
from src.benchmark import PatternGenerator, SearchProfiler
from src.algorithms import BruteForceSearch, BoyerMooreSearch
from src.utils import ResultAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Força Bruta vs Boyer-Moore em SMS Spam"
    )
    parser.add_argument("--data-path",   type=str, required=True,
                        help="Caminho para o CSV do dataset")
    parser.add_argument("--pattern-sizes", type=int, nargs='+', default=[5,10,20,50],
                        help="Tamanhos de padrões a gerar (aleatório)")
    parser.add_argument("--samples-per-size", type=int, default=10,
                        help="Quantidade de padrões aleatórios por tamanho")
    parser.add_argument("--search-reps",  type=int, default=10,
                        help="Repetições de busca para estatísticas")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Seed para aleatoriedade reproduzível")
    parser.add_argument("--fixed-pattern", type=str, default=None,
                        help="Se passado, usa apenas este padrão fixo")
    parser.add_argument("--output",       type=str, default="benchmark_results.csv",
                        help="Arquivo CSV de saída")
    parser.add_argument(
        "--classify", action="store_true",
        help="Executa modo de detecção: classifica SMS e exibe métricas."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=10,
        help="Quantidade de padrões a extrair para a detecção."
    )
    parser.add_argument(
        "--threshold", type=int, default=1,
        help="Número mínimo de padrões para classificar como spam."
    )
    parser.add_argument(
        "--custom-patterns", nargs='+',
        help=(
            "Lista de padrões fixos para detectar spam "
            "(ex: --custom-patterns free click money)"
        )
    )
    args = parser.parse_args()

    messages = DataLoader.load_sms_data(args.data_path)

    if args.classify:
        from src.data.vocab_builder import build_spam_vocab
        from src.classifier.spam_classifier import SpamClassifier
        from src.utils.metrics import compute_confusion_matrix, precision, recall, f1_score
        from src.benchmark.profiler import SearchProfiler as ClassifierProfiler

        df = pd.read_csv(
            args.data_path,
            encoding="latin1",
            usecols=[0,1],
            names=["label","text"],
            header=0
        )
        labels = df["label"].str.lower().tolist()
        texts = df["text"].str.lower().tolist()

        if args.custom_patterns:
            patterns = [p.lower() for p in args.custom_patterns]
            print(f"Usando padrões customizados: {patterns}")
        else:
            patterns = build_spam_vocab(labels, texts, args.vocab_size)
            print(f"Padrões extraídos do dataset: {patterns}")

        algorithms = {
            "bf": BruteForceSearch(),
            "bm": BoyerMooreSearch()
        }

        if os.path.exists("detection_reports"):
            shutil.rmtree("detection_reports")
        os.makedirs("detection_reports")

        perf_stats = {}

        for name, alg in algorithms.items():
            print(f"\n=== Algoritmo: {name} ===")
            classifier = SpamClassifier(patterns, alg, args.threshold)

            preds = [classifier.classify(t) for t in texts]
            TP, FP, TN, FN = compute_confusion_matrix(labels, preds)
            prec = precision(TP, FP)
            rec  = recall(TP, FN)
            f1   = f1_score(prec, rec)

            print("— Acurácia —")
            print(f" TP={TP}, FP={FP}, TN={TN}, FN={FN}")
            print(f" Precision: {prec:.3f}, Recall: {rec:.3f}, F1-score: {f1:.3f}")

            stats = ClassifierProfiler.benchmark_classification(
                classifier, texts, args.search_reps
            )
            perf_stats[name] = stats

            print("— Performance da classificação —")
            print(f" Tempo médio total: {stats['mean_s']:.3f} s")
            print(f" Stdev total:       {stats['stdev_s']:.3f} s")
            print(f" Min total:         {stats['min_s']:.3f} s")
            print(f" Max total:         {stats['max_s']:.3f} s")
            print(f" p90 total:         {stats['p90_s']:.3f} s")
            print(f" p95 total:         {stats['p95_s']:.3f} s")
            print(f" Tempo médio/mensagem: {stats['mean_per_msg_ms']:.3f} ms")
            print(f" p90/mensagem:         {stats['p90_per_msg_ms']:.3f} ms")
            print(f" p95/mensagem:         {stats['p95_per_msg_ms']:.3f} ms")
            print(f" Uso de memória (aprox.): {stats['space_bytes']} bytes")

            details = []
            for true_label, text in zip(labels, texts):
                matched = []
                lower_text = text.lower()
                for pat in patterns:
                    if isinstance(alg, BoyerMooreSearch):
                        alg.preprocess(pat)
                    if alg.search(lower_text, pat) != -1:
                        matched.append(pat)
                decision = "spam" if len(matched) >= args.threshold else "ham"
                details.append({
                    "text": text,
                    "true_label": true_label,
                    "predicted": decision,
                    "match_count": len(matched),
                    "matched_patterns": ";".join(matched)
                })
            df_details = pd.DataFrame(details)

            all_path = f"detection_reports/details_{name}.csv"
            df_details.to_csv(all_path, index=False)
            print(f"Gravado relatório completo em {all_path}")

            spam_only = df_details[df_details["predicted"] == "spam"]
            spam_path = f"detection_reports/spam_only_{name}.csv"
            spam_only.to_csv(spam_path, index=False)
            print(f"Gravado apenas spams em {spam_path}")

        labels_alg = list(perf_stats.keys())
        mean_times = [perf_stats[n]['mean_s'] for n in labels_alg]
        memory_usage = [perf_stats[n]['space_bytes'] for n in labels_alg]
        per_msg_times  = [perf_stats[n]['mean_per_msg_ms'] for n in labels_alg]

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle("Comparação: Força Bruta vs Boyer-Moore - Classificação de Spam", fontsize=16, fontweight='bold')

        axs[0].bar(labels_alg, mean_times, color=['blue', 'green'])
        axs[0].set_ylabel('Tempo de Execução (s)')
        axs[0].set_title('Tempo médio de classificação por algoritmo')

        axs[1].bar(labels_alg, memory_usage)
        axs[1].set_ylabel('Uso de memória (bytes)')
        axs[1].set_title('Memória alocada para classificação por algoritmo')

        axs[2].plot(
            labels_alg,
            per_msg_times,
            marker='o',
            linestyle='-',
            label='Tempo/mensagem (ms)'
        )
        axs[2].set_title("Tempo médio por mensagem (ms)")
        axs[2].set_xlabel("Algoritmo")
        axs[2].set_ylabel("Tempo por mensagem (ms)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.5)
        plt.show()

        return

    if args.fixed_pattern:
        patterns_map = {len(args.fixed_pattern): [args.fixed_pattern.lower()]}
    else:
        patterns_map = PatternGenerator.generate_patterns(
            messages,
            args.pattern_sizes,
            args.samples_per_size,
            args.seed
        )

    algorithms = {
        "bf": BruteForceSearch(),
        "bm": BoyerMooreSearch()
    }
    rows = []

    for size, pats in patterns_map.items():
        for pat in pats:
            build_time_ms = 0.0
            if isinstance(algorithms["bm"], BoyerMooreSearch):
                start_build = time.perf_counter()
                algorithms["bm"].preprocess(pat)
                build_time_ms = (time.perf_counter() - start_build) * 1e3

            results = {}
            for algo_name, algorithm in algorithms.items():
                stats = SearchProfiler.benchmark(
                    algorithm,
                    messages,
                    pat,
                    args.search_reps
                )
                results[algo_name] = stats

            speedup = (
                results["bf"]["mean_s"] / results["bm"]["mean_s"]
                if results["bm"]["mean_s"] > 0 else 0
            )

            row = {
                "pattern_size": size,
                "pattern": pat,
                "build_time_ms": build_time_ms,
                "speedup": speedup
            }
            for algo_name, stats in results.items():
                for k, v in stats.items():
                    row[f"{algo_name}_{k}"] = v
            rows.append(row)

    ResultAnalyzer.save_results(rows, args.output)


if __name__ == "__main__":
    main()
