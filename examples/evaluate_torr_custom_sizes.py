from unitxt import evaluate, load_dataset, settings
from unitxt.benchmark import Benchmark
from unitxt.inference import (
    CrossProviderInferenceEngine,
)

benchmark = Benchmark(
    subsets={
        "fin_qa": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.fin_qa.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.fin_qa.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.fin_qa.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.fin_qa.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.fin_qa.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "numeric_nlg": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.numeric_nlg.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.numeric_nlg.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.numeric_nlg.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.numeric_nlg.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "qtsumm": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.qtsumm.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.qtsumm.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.qtsumm.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.qtsumm.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.qtsumm.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "scigen": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.scigen.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.scigen.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.scigen.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.scigen.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.scigen.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "tab_fact": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tab_fact.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tab_fact.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tab_fact.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tab_fact.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tab_fact.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "tablebench_data_analysis": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_data_analysis.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "tablebench_fact_checking": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_fact_checking.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "tablebench_numerical_reasoning": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.tablebench_numerical_reasoning.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "turl_col_type": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.concat.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.concat.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.concat.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.concat.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.concat.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.csv.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.csv.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.csv.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.csv.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.csv.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.df.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.df.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.df.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.df.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.df.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.html.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.html.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.html.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.html.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.html.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.indexed_row_major.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.indexed_row_major.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.indexed_row_major.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.indexed_row_major.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.indexed_row_major.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.json.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.json.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.json.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.json.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.json.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_5_demos": "recipes.torr.turl_col_type.markdown.insert_empty_rows_augmentation_5_demos",
                        "no_augmentation_5_demos": "recipes.torr.turl_col_type.markdown.no_augmentation_5_demos",
                        "shuffle_cols_augmentation_5_demos": "recipes.torr.turl_col_type.markdown.shuffle_cols_augmentation_5_demos",
                        "shuffle_rows_augmentation_5_demos": "recipes.torr.turl_col_type.markdown.shuffle_rows_augmentation_5_demos",
                        "transpose_augmentation_5_demos": "recipes.torr.turl_col_type.markdown.transpose_augmentation_5_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
        "wikitq": Benchmark(
            subsets={
                "concat": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.concat.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.concat.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.concat.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.concat.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.concat.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "csv": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.csv.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.csv.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.csv.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.csv.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.csv.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "df": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.df.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.df.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.df.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.df.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.df.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "html": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.html.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.html.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.html.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.html.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.html.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "indexed_row_major": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.indexed_row_major.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.indexed_row_major.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.indexed_row_major.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.indexed_row_major.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.indexed_row_major.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "json": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.json.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.json.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.json.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.json.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.json.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
                "markdown": Benchmark(
                    subsets={
                        "insert_empty_rows_augmentation_1_demos": "recipes.torr.wikitq.markdown.insert_empty_rows_augmentation_1_demos",
                        "no_augmentation_1_demos": "recipes.torr.wikitq.markdown.no_augmentation_1_demos",
                        "shuffle_cols_augmentation_1_demos": "recipes.torr.wikitq.markdown.shuffle_cols_augmentation_1_demos",
                        "shuffle_rows_augmentation_1_demos": "recipes.torr.wikitq.markdown.shuffle_rows_augmentation_1_demos",
                        "transpose_augmentation_1_demos": "recipes.torr.wikitq.markdown.transpose_augmentation_1_demos",
                    },
                    max_samples_per_subset=100,
                ),
            },
        ),
    },
)

with settings.context(
    allow_unverified_code=True,
):
    test_dataset = load_dataset(
        benchmark,
        split="test",
        use_cache=True,
    )
    # Infer
    model = CrossProviderInferenceEngine(
        model="llama-3-8b-instruct",
        max_tokens=512,
    )
    """
    We are using a CrossProviderInferenceEngine inference engine that supply api access to provider such as:
    watsonx, bam, openai, azure, aws and more.

    For the arguments these inference engines can receive, please refer to the classes documentation or read
    about the the open ai api arguments the CrossProviderInferenceEngine follows.
    """

    predictions = model(test_dataset)
    results = evaluate(predictions=predictions, data=test_dataset)

    print("Global scores:")
    print(results.global_scores.summary)
    print("Subsets scores:")
    print(results.subsets_scores.summary)
