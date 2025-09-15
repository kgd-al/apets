import argparse
import itertools
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from pandas import Series
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
parser.add_argument("--synthesis", default=False, action="store_true", help="Only produce synthesis plots")
args = parser.parse_args()

# ==============================================================================

training_curves_file = args.root.joinpath("training_curves.pdf")
if not args.synthesis and (args.purge or not training_curves_file.exists()):
    print("Hello")
    print("Bye")
    from tbparse import SummaryReader

    log_dir = "<PATH_TO_EVENT_FILE_OR_DIRECTORY>"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    print(df)
    exit(42)


# ==============================================================================

df_file = args.root.joinpath("summaries.csv")
if args.purge and df_file.exists():
    df_file.unlink()

if df_file.exists():
    df = pd.read_csv(df_file, index_col=0)

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

    df["instability_avg"] = df[["avg_roll", "avg_pitch"]].abs().max(axis=1)
    df["instability_std"] = df[["std_roll", "std_pitch"]].max(axis=1)

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
    for r in rewards:
        index = (df.reward == r)
        df.loc[index, "normalized_reward"] = StandardScaler().fit_transform(np.array(df.loc[index, r]).reshape(-1, 1))

    df.to_csv(df_file)

# ==============================================================================

col_mapping = {}
groups = col_mapping["groups"] = "Groups"
all_groups = col_mapping["detailed-groups"] = "Detailed groups"
params = col_mapping["params"] = "Parameters"
reward = col_mapping["reward"] = "Reward"
normal_reward = col_mapping["normalized_reward"] = "Normalized Reward"
kernels_reward = col_mapping["kernels"] = "Gaussians"
lazy_reward = col_mapping["lazy"] = "Gym-Ant"
speed_reward = col_mapping["speed"] = "Speed"
instability_avg = col_mapping["instability_avg"] = "Instability (avg)"
instability_std = col_mapping["instability_std"] = "Instability (std)"
df.rename(inplace=True, columns=col_mapping)

df[reward] = df[reward].map(lambda x: col_mapping[x])


def _groups(_detailed): return all_groups if _detailed else groups


groups_hue_order = sorted(df[groups].unique().tolist())
detailed_groups_hue_order = sorted(df[all_groups].unique().tolist())

rewards = df[reward].unique().tolist()
rewards_hue_order = sorted(df[reward].unique().tolist())


def hue_order(_detailed):
    return detailed_groups_hue_order if _detailed else groups_hue_order


print(rewards)
print(df.columns)
print(df)

# print()
# print(df.groupby(df.index.map(lambda _s: "/".join(_s.split('/')[1:4]))).size().to_string(max_rows=1000))
# print()

print()
print("Bests")
columns = [reward, "arch", "neighborhood", "width", "depth",
           kernels_reward, speed_reward, lazy_reward,
           groups, all_groups]
champs = df.loc[pd.concat(
    df[df[reward] == r].groupby(all_groups, dropna=False)[r].idxmax()
    for r in rewards
)][columns]
champs["SUM"] = champs[speed_reward] + champs[kernels_reward]
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


ss_pareto = _pareto(np.array([(a, b) for a, b in zip(df[speed_reward], df[kernels_reward])]))
ss_pareto = df.iloc[ss_pareto][columns]

print("Speed vs Stability pareto front")
print(ss_pareto)
print()
print(ss_pareto.groupby([reward]).size())
print()
print(ss_pareto.groupby([all_groups]).size())
print()
print(ss_pareto.groupby([reward, all_groups]).size())
print()

pareto_folder = args.root.joinpath("showcase").joinpath("pareto")
if args.purge:
    shutil.rmtree(pareto_folder, ignore_errors=True)
if not pareto_folder.exists():
    pareto_folder.mkdir(parents=True)
    for p in ss_pareto.index:
        showcase(p, pareto_folder)
    print()


print("Performance vs energy pareto front")
print()

pe_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(1 - df["avg_d_o"], df[normal_reward])]))]
print(pe_pareto[columns])

print()
print("Pareto front:")
print()
print(pe_pareto.groupby([reward]).size())
print()
print(pe_pareto.groupby([all_groups]).size())
print()
print(pe_pareto.groupby([reward, all_groups]).size())
print()


# ==============================================================================

trajectories_file = args.root.joinpath("trajectories.pdf")
if not args.synthesis and (args.purge or not trajectories_file.exists()):
    with PdfPages(trajectories_file) as summary_pdf:
        fig, ax = plt.subplots()
        tdfs_trajs = {}

        sns_cp = sns.color_palette()
        for f in tqdm.tqdm(
                args.root.glob("**/trajectory.csv"),
                desc="Processing"):
            tdf = pd.read_csv(f, index_col=0)
            run = str(f.parent)
            data = df.loc[run, :]
            sns.lineplot(data=tdf, x="x", y="y", ax=ax,
                         color=sns_cp[groups_hue_order.index(data[groups])],
                         zorder=-tdf.x.iloc[-1], lw=.1)

            tdfs_trajs[run] = tdf

        ax.legend(handles=[
            Line2D([0], [0], color=sns_cp[i], label=groups_hue_order[i])
            for i in range(len(groups_hue_order))
        ])

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        for r, tdf in tdfs_trajs.items():
            data = df.loc[r, :]
            sns.lineplot(data=tdf, x="x", y="y", ax=ax,
                         color=sns_cp[rewards_hue_order.index(data[reward])],
                         zorder=-tdf.x.iloc[-1], lw=.1)

        ax.legend(handles=[
            Line2D([0], [0], color=sns_cp[i], label=rewards_hue_order[i])
            for i in range(len(rewards_hue_order))
        ])

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        sns_cp = sns.color_palette(n_colors=len(champs))
        for r in champs.index:
            sns.lineplot(data=tdfs_trajs[r], x="x", y="y", ax=ax,)

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        sns_cp = sns.color_palette(n_colors=len(champs))
        for r in champs.index:
            sns.lineplot(data=tdfs_trajs[r], x="t", y="z", ax=ax,)

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

# ==============================================================================

def __key(*__args): return "_".join([str(x) for x in __args])


synthesis = defaultdict(list)
synthesis.update({
    sns.relplot: [
        __key(params, r, False) for r in rewards
    ]
})
print(synthesis)


def is_synthesis(fn, _args):
    r = __key(*_args) in synthesis.get(fn, [])
    print(__key(*_args), "->", r)
    return r


def maybe_save(_g, _is_synthesis):
    if not args.synthesis:
        summary_pdf.savefig(_g.figure, bbox_inches="tight")
    if _is_synthesis:
        synthesis_pdf.savefig(_g.figure, bbox_inches="tight")
    plt.close()


pdf_summary_file = args.root.joinpath("summary.pdf")
pdf_synthesis_file = args.root.joinpath("synthesis.pdf")
print("Plotting...")
sns.set_style("darkgrid")
with PdfPages(pdf_summary_file) as summary_pdf, PdfPages(pdf_synthesis_file) as synthesis_pdf:
    # g = sns.scatterplot(df, x=params, y="depth")
    # plt.xscale('log', base=10)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()
    #
    # g = sns.scatterplot(df, x="width", y="depth")
    # plt.xscale('log', base=2)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()

    def relplot(x, y, base=10, **kwargs):
        _detailed = kwargs.pop("detailed", False)
        if (_isS := kwargs.pop("synthesis", None)) is None:
            _isS = is_synthesis(sns.relplot, (x, y, _detailed))
        if not args.synthesis or _isS:
            _args = dict(x=x, y=y,
                         hue=_groups(_detailed), hue_order=hue_order(_detailed),
                         col=reward,
                         kind='line', marker='o',
                         err_style="bars", errorbar="ci", estimator="median")

            _args.update(kwargs)
            g = sns.relplot(df, **_args)
            plt.xscale('log', base=base)
            maybe_save(g, _isS)

    pareto_args = dict(
        linestyle='dashed', color="red", lw=.5,
        marker='D', markeredgecolor='k', markersize=7,
        label="Pareto front", zorder=5
    )

    for detailed in [False, True]:
        # Speed vs kernels + pareto front

        isS = is_synthesis(sns.scatterplot, (speed_reward, kernels_reward, detailed))
        if not args.synthesis or isS:
            scatter_args = dict(
                x=speed_reward, y=kernels_reward, style=reward,
                hue=_groups(detailed), hue_order=hue_order(detailed),
            )

            g = sns.scatterplot(df, **scatter_args)
            g.axes.plot(ss_pareto[speed_reward], ss_pareto[kernels_reward], **pareto_args)
            sns.scatterplot(ss_pareto, **scatter_args, ax=g.axes, zorder=10, legend=False)

            g.axes.legend()
            maybe_save(g, isS)

        # ===
        # = Energy vs performance + pareto front

        isS = is_synthesis(sns.scatterplot, ("avg_d_o", normal_reward, detailed))
        if not args.synthesis or isS:
            scatter_args = dict(
                x="avg_d_o", y=normal_reward, style=reward,
                hue=_groups(detailed), hue_order=hue_order(detailed),
            )

            g = sns.scatterplot(df, **scatter_args)
            g.axes.plot(pe_pareto["avg_d_o"], pe_pareto[normal_reward], **pareto_args)
            sns.scatterplot(pe_pareto, **scatter_args, ax=g.axes, zorder=10, legend=False)

            g.legend().remove()
            g.figure.legend(loc='outside right')
            maybe_save(g, isS)

        # ====

        for k in rewards:
            relplot(x=params, y=k, detailed=detailed, synthesis=False)
            relplot(x=params, y=k, errorbar=("pi", 100),
                    err_style="band", detailed=detailed, synthesis=False)
    print()

    tested_pairs = [
        # ((a, b), (c, d)) for ((a, b), (c, d)) in
        ((b, a), (d, c)) for ((a, b), (c, d)) in
        itertools.combinations(itertools.product(df[groups].unique(), df[reward].unique()), r=2)
        if a == c or b == d
    ]
    tests = pd.DataFrame(index=tested_pairs, columns=[])
    if len(synthesis[sns.violinplot]) > 0:
        print("Testing", len(tested_pairs), "pairs:", tested_pairs)
    annotator = None

    relplot(x=params, y="tps")
    # for c in ["avg_d_o", "Vx", "Vy", "Vz", "z", "dX", "dY", "cX", "cY"] + [c for c in df.columns if ("avg" in c or "std" in c)]:
    for c in rewards + [c for c in df.columns if ("avg" in c or "std" in c)]:
        relplot(x=params, y=c)

        # ===
        isS = is_synthesis(sns.violinplot, (c,))
        if not args.synthesis or isS:
            violinplot_args = dict(
                # data=df, x=groups, y=c, hue=reward,
                data=df, x=reward, y=c, hue=groups,
                inner="box", cut=0, gap=.25,
                common_norm=True, density_norm="width"
            )

            ax = sns.violinplot(**violinplot_args)

            # print(c)
            # annotator = Annotator(ax=ax, pairs=tested_pairs, plot='violinplot', **violinplot_args)
            # annotator.configure(test="Mann-Whitney", verbose=2,
            #                     hide_non_significant=True, text_format="simple",
            #                     comparisons_correction="bonferroni")
            # _, corrected_results = annotator.apply_and_annotate()
            # tests[c] = [r.data.pvalue for r in corrected_results]
            # print()
            maybe_save(ax, isS)

            violinplot_args["x"], violinplot_args["hue"] = violinplot_args["hue"], violinplot_args["x"]
            ax = sns.violinplot(**violinplot_args)

            maybe_save(ax, isS)

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

    if tests.size > 0:
        tests = tests.map(lambda x: x if x <= 0.05 else np.nan)
        g = sns.heatmap(tests, annot=True, cmap="magma_r", fmt=".2g", annot_kws=dict(size=3), norm=LogNorm())
        summary_pdf.savefig(g.figure, bbox_inches="tight")
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

    if not args.synthesis:
        for df_name, base_df in [("all champions", champs)]:#, ("the pareto front", pareto)]:
            df = pd.concat([_process(f) for f in base_df.index])
            df.set_index(["Path", "m"], inplace=True)

            for label, value in [("Amplitude", "A"), ("Frequency", "f"), ("Phase", "p"), ("Offset", "c")]:
                ax = sns.stripplot(df[df.Valid], x=value, y="m", hue="Path")
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                ax.set_title(f"{label} distributions per joint for {df_name}")
                summary_pdf.savefig(ax.figure, bbox_inches="tight")
                plt.close()


print("Generated", pdf_summary_file)
