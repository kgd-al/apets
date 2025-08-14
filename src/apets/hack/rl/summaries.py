import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser("Sumarizes summary.csv files")
parser.add_argument("root", type=Path, nargs="+")
args = parser.parse_args()

df = pd.concat(
    pd.read_csv(f, index_col=0)
    for root in args.root
    for f in root.glob("**/summary.csv")
)

print(df.columns)
print(df)

sns.set_style("darkgrid")
pdf_file = args.root[0].joinpath("summary.pdf")
with PdfPages(pdf_file) as pdf:
    g = sns.scatterplot(df, x="params", y="depth")
    plt.xscale('log', base=10)
    pdf.savefig(g.figure, bbox_inches="tight")
    plt.close()

    g = sns.scatterplot(df, x="width", y="depth")
    plt.xscale('log', base=2)
    pdf.savefig(g.figure, bbox_inches="tight")
    plt.close()

    def hue(overview):
        def fmt_rl(_s):
            trainer = _s.split("/")[4]
            if trainer == "rlearn":
                trainer = "ppo"
            return trainer
        r = df.arch + "-" + df.index.map(fmt_rl)
        if not overview:
            r += df.depth.map(lambda f: str(int(f)) if not np.isnan(f) else "")
        return r

    def plot(x, y, base=10, **kwargs):
        _args = dict(x=x, y=y, hue=hue(kwargs.pop("overview", False)), col="reward",
                     kind='line', marker='o',
                     err_style="bars", errorbar="sd")

        _args.update(kwargs)
        g = sns.relplot(df, **_args)
        plt.xscale('log', base=base)

        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    plot(x="params", y="speed", overview=True)
    plot(x="params", y="speed", errorbar=("pi", 100), err_style="band", overview=True)
    plot(x="params", y="speed")
    plot(x="params", y="speed", errorbar=("pi", 100), err_style="band")
    # plot(x="width", y="speed", base=2)
    plot(x="params", y="tps")
    # plot(x="width", y="tps", base=2)

print("Generated", pdf_file)
