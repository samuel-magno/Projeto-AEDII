import argparse
import time
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
    args = parser.parse_args()

    messages = DataLoader.load_sms_data(args.data_path)

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

            speedup = results["bf"]["mean_s"] / results["bm"]["mean_s"] if results["bm"]["mean_s"] > 0 else 0
            
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