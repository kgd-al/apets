import argparse
import itertools
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas import Series
from statannotations.Annotator import Annotator

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
args = parser.parse_args()

df = pd.concat(
    pd.read_csv(f, index_col=0)
    for f in args.root.glob("**/summary.csv")
)

str_root = str(args.root)
df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str_root))
print(df.columns)

invalid_runs = [
    f"cma/{r}/mlp-{w}"
    for r in ["kernels", "distance"]
    for w in ["2-128", "2-64", "1-128"]
]
print("Dropping invalid runs:", " ".join(invalid_runs))

df = df[~df.index.map(lambda _s: any(__s in _s for __s in invalid_runs))]
print(df)

df["|avg_y|"] = df["avg_y"].abs()

# print()
# print(df.groupby(df.index.map(lambda _s: "/".join(_s.split('/')[1:4]))).size().to_string(max_rows=1000))
# print()

print()
print("Bests")
arch_groups = df.groupby(["arch", "depth"], dropna=False)
columns = ["arch", "neighborhood", "width", "depth", "kernels", "speed"]
champs = df.loc[pd.concat([arch_groups.kernels.idxmax(), arch_groups.speed.idxmax()])][columns]
champs["SUM"] = champs["speed"] + champs["kernels"]
champs.sort_values(inplace=True, by="SUM", ascending=False)
print(champs)
print()


def showcase(_p, _out):
    def cp(_src):
        _dst = _out.joinpath(Path(str(_src.parent).replace(str_root + "/", "").replace("/", "_")).with_suffix(_src.suffix))
        print(_src, "->", _dst)
        shutil.copyfile(_src, _dst)

    _p = Path(_p)
    cp(_p.joinpath("movie.mp4"))
    cp(_p.joinpath("trajectory.pdf"))


best_folder = args.root.joinpath("showcase").joinpath("bests")
if args.purge:
    shutil.rmtree(best_folder, ignore_errors=True)
if not best_folder.exists():
    best_folder.mkdir(parents=True)
    for p in champs.index:
        showcase(p, best_folder)
    print()


def _pareto(_points):
    _original_points = _points.copy()
    # Credit goes to https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    is_efficient = np.arange(_points.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(_points):
        nondominated_point_mask = np.any(_points > _points[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        _points = _points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    return sorted(is_efficient, key=lambda i: math.atan2(_original_points[i][1], _original_points[i][0]))


pareto = _pareto(np.array([(a, b) for a, b in zip(df["speed"], df["kernels"])]))
pareto = df.iloc[pareto][columns]

print("Pareto front")
print(pareto)
print()

pareto_folder = args.root.joinpath("showcase").joinpath("pareto")
if args.purge:
    shutil.rmtree(pareto_folder, ignore_errors=True)
if not pareto_folder.exists():
    pareto_folder.mkdir(parents=True)
    for p in pareto.index:
        showcase(p, pareto_folder)
    print()


print("...")


def _hue(_overview=True):
    def fmt_rl(_s):
        trainer = _s.split("/")[-4]
        if trainer == "rlearn":
            trainer = "ppo"
        return trainer

    return Series(name="arch" + ("" if _overview else "-detailed"),
                  data=(
                          df.arch
                          + ("" if _overview else df.depth.map(lambda f: str(int(f)) if not np.isnan(f) else ""))
                          + "-" + df.index.map(fmt_rl)
                  ))


hues = {_overview: _hue(_overview=_overview) for _overview in [True, False]}
df["groups"] = groups = hues[True]
df["detailed-groups"] = detailed_groups = hues[False]

print("Plotting...")

sns.set_style("darkgrid")
pdf_file = args.root.joinpath("summary.pdf")
with (PdfPages(pdf_file) as pdf):
    # g = sns.scatterplot(df, x="params", y="depth")
    # plt.xscale('log', base=10)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()
    #
    # g = sns.scatterplot(df, x="width", y="depth")
    # plt.xscale('log', base=2)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()

    def plot(x, y, base=10, **kwargs):
        _args = dict(x=x, y=y,
                     hue=hues[kwargs.pop("overview", True)],
                     col="reward",
                     kind='line', marker='o',
                     err_style="bars", errorbar="ci", estimator="median")

        _args.update(kwargs)
        g = sns.relplot(df, **_args)
        plt.xscale('log', base=base)

        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    for overview in [True, False]:
        g = sns.scatterplot(df, x="speed", y="kernels", hue=hues[overview], style="reward")
        g.axes.plot(pareto.speed, pareto.kernels, 'r--', lw=.5, zorder=-10, marker='D',
                    label="Pareto front")
        g.axes.legend()

        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        for k in ["speed", "kernels"]:
            plot(x="params", y=k, overview=overview)
            plot(x="params", y=k, errorbar=("pi", 100), err_style="band", overview=overview)
    print()

    tested_pairs = [
        ((a, b), (c, d)) for ((a, b), (c, d)) in
        itertools.combinations(itertools.product(df["groups"].unique(), df["reward"].unique()), r=2)
        if a == c or b == d
    ]
    print("Testing", len(tested_pairs), "pairs:", tested_pairs)
    annotator = None

    plot(x="params", y="tps")
    for c in ["Vx", "Vy", "Vz", "z", "dX", "dY", "cX", "cY"] + [c for c in df.columns if ("avg" in c or "std" in c)]:
        print(c)
        plot(x="params", y=c)

        # ===
        violinplot_args = dict(
            data=df, x="groups", y=c, hue="reward",
            split=True, gap=.1, inner="quart", cut=0,
            common_norm=True, density_norm="area"
        )

        ax = sns.violinplot(**violinplot_args)

        try:
            annotator = Annotator(ax=ax, pairs=tested_pairs, plot='violinplot', **violinplot_args)
            annotator.configure(test="Mann-Whitney", verbose=2,
                                hide_non_significant=True, text_format="simple",
                                comparisons_correction="bonferroni")
            annotator.apply_and_annotate()
            print()
        except:
            print(r"/!\\/!\\")

        pdf.savefig(ax.figure, bbox_inches="tight")
        plt.close()

        # ===
        fig, ax = plt.subplots()
        sub_violinplots_args = violinplot_args.copy()
        sub_violinplots_args["inner"] = None
        handles, labels = [], []
        for (hue, group), color in zip(
                detailed_groups.groupby(detailed_groups, sort=False),
                sns.color_palette(None, detailed_groups.unique().size)):
            sub_violinplots_args["data"] = df.loc[group.index]
            sub_violinplots_args["x"] = groups[group.index]
            sub_violinplots_args["palette"] = (color, .5 + .5 * np.array(color))
            sns.violinplot(**sub_violinplots_args, alpha=.25, legend=False)
            handles.append(plt.Rectangle((0, 0), 0, 0,
                                         fc=color))
            labels.append(hue)

        fig.legend(
            labels=labels, handles=handles,
            title="arch-detailed",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        # ===


    def _process(_path):
        _path = Path(_path)
        _df = pd.read_csv(_path.joinpath("motors.csv"))
        _df["Path"] = _path
        _df["Valid"] = _df.f.map(lambda x: .1 <= x <= 10)
        return _df

    for df_name, base_df in [("all champions", champs)]:#, ("the pareto front", pareto)]:
        df = pd.concat([_process(f) for f in base_df.index])
        df.set_index(["Path", "m"], inplace=True)
        print(df)

        for label, value in [("Amplitude", "A"), ("Frequency", "f"), ("Phase", "p"), ("Offset", "c")]:
            ax = sns.stripplot(df[df.Valid], x=value, y="m", hue="Path")
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.set_title(f"{label} distributions per joint for {df_name}")
            pdf.savefig(ax.figure, bbox_inches="tight")
            plt.close()


print("Generated", pdf_file)
