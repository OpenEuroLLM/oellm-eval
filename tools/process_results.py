import numpy as np
import logging
import argparse
import os
from os import path
import json
import matplotlib.pyplot as plt

# Mapping from task groups to aggregated metrics
mapping = {
    "nob_sentiment_analysis_cf": "f1,none",
    "nob_sentiment_analysis_hybrid": "f1,none",
    "nob_sentiment_analysis_mcf": "f1,none",
    "nob_reading": "f1,none",
    "nob_instruction": "bleu,none",
    "nob_mt": "bleu,none",
    "nob_language": "fscore,none",
    "nob_generation": "rougeL_max,none",
    "nob_world_cf": "acc,none",
    "nob_world_hybrid": "acc,none",
    "nob_world_mcf": "acc,none",
    "nob_orthography": "acc,none",
}

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--results", "-r", help="Path to the NorEval results directory "
                                "(should contain subdirectories corresponding to task groups)", required=True)
    arg(
        "--benchmark",
        "-b",
        help="Benchmarks to process, split by comma (e.g., 'nob_sentiment_analysis_mcf,nob_instruction')",
        required=True,
    )
    arg("--shots", "-s", help="Number of shots", default=0, type=int)
    arg("--plot", "-p", help="Name of the file to save the plot")

    args = parser.parse_args()

    benchmark_labels = args.benchmark.split(",")

    results = {}
    scoreboard = {}

    # Parsing jsons, loading the data from lm_eval
    for benchmark in benchmark_labels:
        logger.info(benchmark)
        full_path = path.join(args.results, benchmark, f"{args.shots}-shot")
        for model in os.scandir(full_path):
            logger.info(f"Processing {model.name}...")
            with os.scandir(path.join(full_path, model.name)) as el:
                for entry in el:
                    if entry.name.endswith(".json") and entry.is_file():
                        logger.info(f"Loading {entry.name}...")
                        with open(entry.path) as f:
                            data = json.load(f)
                        short_model = model.name.split("__")[-2]
                        if short_model not in results:
                            results[short_model] = {}
                        results[short_model][benchmark] = data
                        break

        # Populating the scoreboard
        for model in results:
            if model not in scoreboard:
                scoreboard[model] = []
            for res in results[model][benchmark]["results"]:
                if res == benchmark:
                    metrics = mapping[res]
                    logger.info(
                        f"{model}: {results[model][benchmark]["results"][res][metrics]}"
                    )
                    scoreboard[model].append(
                        results[model][benchmark]["results"][res][metrics]
                    )

    # Plotting the results
    x = np.arange(len(benchmark_labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    plt.rcParams['font.size'] = 6
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=4)

    for attribute, measurement in scoreboard.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Averaged scores")
    ax.set_xlabel("Benchmark groups and metrics")
    ax.set_title(f"NorEval performance in {args.shots}-shot setup")
    ax.set_xticks(
        x + 0.5 * width,
        [f"{el} ({mapping[el].split(",")[0]})" for el in benchmark_labels],
    )
    ax.legend(loc="best", ncols=3)

    if args.plot:
        plt.savefig(args.plot, dpi=200)
        logger.info(f"Plot saved to {args.plot}")

    # Show the chart
    plt.show()
