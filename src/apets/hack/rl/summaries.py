import argparse
import itertools
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from pandas import Series
from statannotations.Annotator import Annotator

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
args = parser.parse_args()

df_file = args.root.joinpath("summaries.csv")
if args.purge and df_file.exists():
    df_file.unlink()

if df_file.exists():
    df = pd.read_csv(df_file, index_col=0)
    rewards = df.reward.unique().tolist()

else:
    df = pd.concat(
        pd.read_csv(f, index_col=0)
        for f in args.root.glob("**/summary.csv")
    )

    str_root = str(args.root)
    df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str_root))

    invalid_runs = [
        f"cma/{r}/mlp-{w}"
        for r in ["kernels", "distance"]
        for w in ["2-128", "2-64", "1-128"]
    ]
    print("Dropping invalid runs:", " ".join(invalid_runs))

    df = df[~df.index.map(lambda _s: any(__s in _s for __s in invalid_runs))]

    df["|avg_y|"] = df["avg_y"].abs()

    def compute_d_o(_path):
        try:
            _df = pd.read_csv(Path(_path).joinpath("joints_data.csv"))
            _df = _df[[c for c in _df.columns if c[-5:] == "-ctrl"]]
            return (_df.iloc[1:].reset_index(drop=True) - _df.iloc[:-1]).abs().mean().mean()
        except FileNotFoundError:
            return np.nan

    df["avg_d_o"] = df.index.map(compute_d_o)

    df.loc[df.reward == "distance", "reward"] = "speed"

    def _make_groups(_overview=True):
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

    df["groups"] = _make_groups(_overview=True)
    df["detailed-groups"] = _make_groups(_overview=False)

    rewards = df.reward.unique().tolist()
    def _normalize(_df): return (_df - _df.min()) / (_df.max() - _df.min())
    for r in rewards:
        index = (df.reward == r)
        df.loc[index, "normalized_reward"] = _normalize(df.loc[index, r])

    df.to_csv(df_file)


def groups(_detailed):
    return df["detailed-groups"] if _detailed else df["groups"]


print(rewards)
print(df.columns)
print(df)

# print()
# print(df.groupby(df.index.map(lambda _s: "/".join(_s.split('/')[1:4]))).size().to_string(max_rows=1000))
# print()

print()
print("Bests")
columns = ["reward", "arch", "neighborhood", "width", "depth", "kernels", "speed", "lazy", "detailed-groups"]
champs = df.loc[pd.concat(
    df[df.reward == r].groupby("detailed-groups", dropna=False)[r].idxmax()
    for r in rewards
)][columns]
champs["SUM"] = champs["speed"] + champs["kernels"]
champs.sort_values(inplace=True, by="SUM", ascending=False)
print(champs)
print()


# for c in champs.index:
#     c = Path(c)
#     b = pd.read_csv(c.joinpath("bricks_data.csv"))
#     fig, ax = plt.subplots()
#     for col in [b for b in b.columns if len(b.split("_")) == 4]:
#         ax.plot(b.t, b[col], label=col)
#     ax.legend()
#     fig.savefig(c.joinpath("steps.pdf"), bbox_inches="tight")
#     fig.savefig(c.joinpath("steps.png"), bbox_inches="tight")
#     plt.close()


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
                     hue=groups(kwargs.pop("detailed", False)),
                     col="reward",
                     kind='line', marker='o',
                     err_style="bars", errorbar="ci", estimator="median")

        _args.update(kwargs)
        g = sns.relplot(df, **_args)
        plt.xscale('log', base=base)

        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    for detailed in [False, True]:
        g = sns.scatterplot(df, x="speed", y="kernels", hue=groups(detailed), style="reward")
        g.axes.plot(pareto.speed, pareto.kernels, 'r--', lw=.5, zorder=-10, marker='D',
                    label="Pareto front")
        g.axes.legend()
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        g = sns.scatterplot(df, x="avg_d_o", y="normalized_reward", hue=groups(detailed), style="reward")
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        for k in rewards:
            plot(x="params", y=k, detailed=detailed)
            plot(x="params", y=k, errorbar=("pi", 100), err_style="band", detailed=detailed)
    print()

    print("Purging ant data")
    df = df[~(df.reward == "ant")]
    rewards.remove("ant")

    tested_pairs = [
        ((a, b), (c, d)) for ((a, b), (c, d)) in
        itertools.combinations(itertools.product(df["groups"].unique(), df["reward"].unique()), r=2)
        if a == c or b == d
    ]
    tests = pd.DataFrame(index=tested_pairs, columns=[])
    print("Testing", len(tested_pairs), "pairs:", tested_pairs)
    annotator = None

    plot(x="params", y="tps")
    # for c in ["avg_d_o", "Vx", "Vy", "Vz", "z", "dX", "dY", "cX", "cY"] + [c for c in df.columns if ("avg" in c or "std" in c)]:
    for c in rewards + [c for c in df.columns if ("avg" in c or "std" in c)]:
        print(c)
        plot(x="params", y=c)

        # ===
        violinplot_args = dict(
            data=df, x="groups", y=c, hue="reward",
            inner="box", cut=0, gap=.25,
            common_norm=True, density_norm="width"
        )

        ax = sns.violinplot(**violinplot_args)

        annotator = Annotator(ax=ax, pairs=tested_pairs, plot='violinplot', **violinplot_args)
        annotator.configure(test="Mann-Whitney", verbose=2,
                            hide_non_significant=True, text_format="simple",
                            comparisons_correction="bonferroni")
        _, corrected_results = annotator.apply_and_annotate()
        tests[c] = [r.data.pvalue for r in corrected_results]
        print()

        pdf.savefig(ax.figure, bbox_inches="tight")
        plt.close()

        # # ===
        # fig, ax = plt.subplots()
        # sub_violinplots_args = violinplot_args.copy()
        # sub_violinplots_args["inner"] = None
        # handles, labels = [], []
        # for (hue, group), color in zip(
        #         detailed_groups.groupby(detailed_groups, sort=False),
        #         sns.color_palette(None, detailed_groups.unique().size)):
        #     sub_violinplots_args["data"] = df.loc[group.index]
        #     sub_violinplots_args["x"] = groups[group.index]
        #     sub_violinplots_args["palette"] = (color, .5 + .5 * np.array(color))
        #     sns.violinplot(**sub_violinplots_args, alpha=.25, legend=False)
        #     handles.append(plt.Rectangle((0, 0), 0, 0,
        #                                  fc=color))
        #     labels.append(hue)
        #
        # fig.legend(
        #     labels=labels, handles=handles,
        #     title="arch-detailed",
        # )
        #
        # pdf.savefig(fig, bbox_inches="tight")
        # plt.close()
        # # ===

    tests = tests.map(lambda x: x if x <= 0.05 else np.nan)
    g = sns.heatmap(tests, annot=True, cmap="magma_r", fmt=".2g", annot_kws=dict(size=3), norm=LogNorm())
    pdf.savefig(g.figure, bbox_inches="tight")
    plt.close()

    def _process(_path):
        try:
            _path = Path(_path)
            _df = pd.read_csv(_path.joinpath("motors.csv"))
            _df["Path"] = _path
            _df["Valid"] = _df.f.map(lambda x: .1 <= x <= 10)
            return _df
        except FileNotFoundError:
            return None

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
