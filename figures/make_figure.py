import matplotlib.pyplot as plt
from typing import List
import numpy as np
from itertools import groupby
from matplotlib.ticker import FixedLocator, FixedFormatter



def save_figure(
        x: List[float],
        y: List[float], 
        labels: List[str],
        colors: List[str],
        markers: List[str],
        task_name: str,
        ylabel: str, 
        title: str,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#F5F5F5')

    all_data = sorted(list(zip(x, y, labels, colors, markers)), key=lambda x: x[3])
    grouped_data = groupby(all_data, key=lambda x: x[3])

    for color, group in grouped_data:
        group = list(group)
        for i, (xi, yi, label, _, marker) in enumerate(group):
            ax.plot(xi, yi, marker, color=color, markersize=18, label=color)
            ax.annotate(label, (xi, yi), fontsize=16, color=color, 
                        xytext=(-100, 17) if i>0 else (-15, -33), textcoords='offset pixels')
        
        for left, right in zip(group[:-1], group[1:]):
            ax.plot([left[0], right[0]], [left[1], right[1]], linestyle=':', color=color)

    ax.set_title(f"{task_name} Performance vs Model Size", fontsize=18)
    ax.set_xlabel("Parameters", fontsize=18)
    ax.set_ylabel(f"{task_name} accuracy", fontsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    ax.set_xscale("log")

    # Remove the frame of the chart
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.savefig(f'{task_name}.png')

def main():
    labels = [
            "Code Llama 7B", "Code Llama 34B", "Llemma 7B",
            "Llemma 34B", "Minerva 8B", "Minerva 62B", "Minerva 540B"
    ]
    colors = ['red', 'red', 'blue', 'blue', 'orange', 'orange', 'orange']
    markers = ['o', 'o', '^', '^', 's', 's', 's']

    x = [7e9, 34e9, 7e9, 34e9, 8e9, 62e9, 540e9]  # Params
    y = [4.4, 11.9, 17.2, 24.1, 14.1, 27.6, 33.6]  # MATH Score

    save_figure(
            x=x, y=y, labels=labels, colors=colors, markers=markers,
            task_name="MATH", ylabel="MATH accuracy", title="MATH Performance vs Model Size"
    )

    x = [7e9, 34e9, 7e9, 34e9, 8e9, 62e9, 540e9]  # Params
    y = [10.5, 29.6, 36.4, 51.5, 16.2, 52.4, 58.8]  # MATH Score

    save_figure(
            x=x, y=y, labels=labels, colors=colors, markers=markers,
            task_name="GSM8k", ylabel="GSM8k accuracy", title="GSM8k Performance vs Model Size"
    )

if __name__=="__main__":
    main()
