import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sympy.codegen.ast import int64

parser = argparse.ArgumentParser("Sumarizes summary.csv files")
parser.add_argument("root", type=Path, nargs="+")
args = parser.parse_args()

df = pd.concat(
    pd.read_csv(f, index_col=0)
    for root in args.root
    for f in root.glob("**/summary.csv")
)

print(df)
print(df[df.depth==0])
df.depth = df.depth.astype(str)
print(df.dtypes)

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

    def plot(x, y, base=10):
        g = sns.relplot(df, x=x, y=y, hue="depth", col="reward",
                        kind='line', err_style="bars", marker='o')
        plt.xscale('log', base=base)
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    plot(x="params", y="speed")
    # plot(x="width", y="speed", base=2)
    plot(x="params", y="tps")
    # plot(x="width", y="tps", base=2)

print("Generated", pdf_file)
