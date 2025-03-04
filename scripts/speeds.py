#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <folder>")
    print(f"       Generates evolution speed graphs")
    exit(1)

root = Path(sys.argv[1])

df = pd.DataFrame(columns=["Exp", "Run", "Time", "Vision", "Duration"])


def _split_config(_str): return _str.split("=")[1][:-1]


def _dehumanize(_str):
    _tokens = _str.replace(",", "").replace(" and", "").split()
    _seconds = 0
    for i in range(0, len(_tokens), 2):
        number, unit = float(_tokens[i]), _tokens[i+1]
        if unit[-1] == 's':
            unit = unit[:-1]
        _seconds += number * {
            "day": 24*60*60,
            "hour": 60*60,
            "minute": 60,
            "second": 1
        }[unit]
    return _seconds


time_re = re.compile(r"Completed evolution in (.*)")

for file in sorted(root.glob("**/log")):
    time, vision, duration = None, None, None
    with open(file) as f:
        for line in reversed(f.readlines()):
            line = line.strip()
            if (match := time_re.search(line)) is not None:
                time = _dehumanize(match.group(1))

            if time is not None:
                if line.startswith("vision"):
                    vision = _split_config(line)
                if line.startswith("simulation_duration"):
                    duration = _split_config(line)

    if time is not None:
        run = int(file.parent.stem.split("-")[1])
        exp = str(file.relative_to(root).parent.parent).replace("-vision", "")
        df.loc[len(df.index)] = [exp, run, time, vision, duration]

print(df.to_string())
g = sns.violinplot(data=df, x="Time", y="Exp", hue="Vision",
                   inner="point", split=True, density_norm="width", cut=0)
g.set(xscale="log")
g.set(xlabel="Time (s)")
g.set(ylabel="Experiment")
g.figure.tight_layout()
g.figure.savefig("remote/speeds.pdf")
