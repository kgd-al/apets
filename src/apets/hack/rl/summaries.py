import argparse
import itertools
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
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
from tbparse import SummaryReader

matplotlib.use("agg")
from apets.hack.hardware import extract_controller

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
parser.add_argument("--synthesis", default=False, action="store_true", help="Only produce synthesis plots")
args = parser.parse_args()

# ==============================================================================

training_curves_file = args.root.joinpath("training_curves.pdf")
if True and not args.synthesis and (args.purge or not training_curves_file.exists()):
    print("Hello")
    for f in args.root.glob("**/run-*"):
        training_curve = f.joinpath("_progress.csv")
        if args.purge or not training_curve.exists():
            if (file := f.joinpath("progress.csv")).exists():
                sub_df = pd.read_csv(file, index_col=0, usecols=[2, 17]).dropna()
                # sub_df = pd.read_csv(file, index_col=0)
                print(sub_df.columns)
                print(sub_df)
                exit(43)

            elif (file := f.joinpath("xrecentbest.dat")).exists():
                print(file)
                sub_df = pd.read_csv(file, sep=' ', usecols=[2, 5])#.dropna()
                sub_df = pd.read_csv(file, sep=' ')
                print(sub_df.columns)
                print(sub_df)
                exit(44)

    # df = df.groupyby()
    print("Bye")
exit(42)


# ==============================================================================

str_root = str(args.root)
df_file = args.root.joinpath("summaries.csv")
if args.purge and df_file.exists():
    df_file.unlink()

showcase_folder = args.root.joinpath("showcase")
if args.purge and showcase_folder.exists():
    shutil.rmtree(showcase_folder)

if df_file.exists():
    df = pd.read_csv(df_file, index_col=0)

else:
    df = pd.concat(
        pd.read_csv(f, index_col=0)
        for f in args.root.glob("**/summary.csv")
    )

    df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str_root))

    invalid_runs = [
        f"cma/{r}/mlp-{w}"
        for r in ["kernels", "distance"]
        for w in ["2-128", "2-64", "1-128"]
    ]
    print("Dropping invalid runs:", " ".join(invalid_runs))

    df = df[~df.index.map(lambda _s: any(__s in _s for __s in invalid_runs))]

    try:
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
        
    except Exception as e:
        print("Ignoring mild error", e)

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

if True:
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

else:
    groups = "groups"
    all_groups = "detailed-groups"
    params = "params"
    reward = "reward"
    normal_reward = "normalized_reward"
    kernels_reward = "kernels"
    lazy_reward = "lazy"
    speed_reward = "speed"
    instability_avg = "instability_avg"
    instability_std = "instability_std"


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


def showcase(_p, _out, _prefix=None):
    def cp(_src, _suffix=None):
        _suffix = _suffix or _src.suffix
        _basename = str(_src.parent).replace(str_root + "/", "").replace("/", "_")
        if _prefix is not None:
            _basename = _prefix + "_" + _basename
        _dst = _out.joinpath(Path(_basename).with_suffix(_suffix))
        print(_src, "->", _dst)
        shutil.copyfile(_src, _dst)

    _p = Path(_p)
    cp(_p.joinpath("movie.mp4"))
    cp(_p.joinpath("trajectory.pdf"))

    try:
        if "cma" in _p.parts:
            archive = _p.joinpath("cma-es.pkl")
        elif "rlearn" in _p.parts:
            archive = _p.joinpath("model.zip")
        else:
            raise RuntimeError
        controller_file = extract_controller.main(argparse.Namespace(
            body="spider", quiet=2, file=archive
        ))
        cp(controller_file, _suffix=".revolve_controller.pkl")
    except Exception as e:
        print(f"\033[90mSkipping failed auto-extract for {_p}: {e}\033[0m")


for _g, _name in [(groups, []), (all_groups, ["detailed"])]:
    print()
    print(" ".join(["Bests"] + [f"({_x})" for _x in _name]))
    columns = [reward, "arch", "neighborhood", "width", "depth",
               kernels_reward, speed_reward, lazy_reward,
               groups, all_groups]
    champs = df.loc[pd.concat(
        df[df[reward] == r].groupby(_g, dropna=False)[r].idxmax()
        for r in rewards
    )][columns]
    champs["SUM"] = champs[speed_reward] + champs[kernels_reward]
    champs.sort_values(inplace=True, by="SUM", ascending=False)
    print(champs)
    print()

    best_folder = showcase_folder.joinpath("_".join(["bests"]+_name))
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


def _showcase_pareto(__pareto_front, __name):
    __folder = args.root.joinpath("showcase").joinpath("pareto").joinpath(__name)
    if args.purge:
        shutil.rmtree(__folder, ignore_errors=True)
    if not __folder.exists():
        __folder.mkdir(parents=True)
        _n = len(__pareto_front)
        _digits = math.ceil(math.log10(_n))
        for i, __p in enumerate(__pareto_front.index):
            showcase(__p, __folder, f"{i:0{_digits}d}")
        print()


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
_showcase_pareto(ss_pareto, "speed_stability")


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
_showcase_pareto(pe_pareto, "performance_energy")


print("Elevation vs jumpiness pareto front")
print()

zz_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(df["avg_z"], 1 - df["std_z"])]))]
print(zz_pareto[columns])

print()
print("Pareto front:")
print()
print(zz_pareto.groupby([reward]).size())
print()
print(zz_pareto.groupby([all_groups]).size())
print()
print(zz_pareto.groupby([reward, all_groups]).size())
print()
_showcase_pareto(zz_pareto, "height_jumpiness")


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
        scatter_args = dict(
            style=reward, style_order=rewards_hue_order,
            hue=_groups(detailed), hue_order=hue_order(detailed),
        )

        for x, y, pareto_df in [
            # Speed vs kernels + pareto front
            (speed_reward, kernels_reward, ss_pareto),
            # = Energy vs performance + pareto front
            ("avg_d_o", normal_reward, pe_pareto),
            # = Elevation vs jumpiness + pareto front
            ("avg_z", "std_z", zz_pareto),
        ]:
            isS = is_synthesis(sns.scatterplot, (x, y, detailed))
            if not args.synthesis or isS:
                scatter_args["x"], scatter_args["y"] = x, y

                g = sns.scatterplot(df, **scatter_args)
                g.axes.plot(pareto_df[x], pareto_df[y], **pareto_args)
                sns.scatterplot(pareto_df, **scatter_args, ax=g.axes, zorder=10, legend=False)

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
