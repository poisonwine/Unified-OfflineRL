import csv
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('.')


COLORS = (
    [   
        '#E52B50', # red
        '#318DE9', # blue
        '#FF7D00', # orange
        # '#E52B50', # red
        '#8D6AB8', # purple
        '#00CD66', # green
        '#7f7f7f',  # middle gray
        '#2E8B57', # seagreen
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        # '#7f7f7f',  # middle gray
    ]
)


def merge_csv(root_dir, query_file, query_x, query_y):
    """Merge result in csv_files into a single csv file."""
    csv_files = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            if f == query_file:
                csv_files.append(os.path.join(dirname, f))
    results = {}
    for csv_file in csv_files:
        content = [[query_x, query_y]]
        df = pd.read_csv(csv_file)
        values = df[[query_x, query_y]].values
        print(len(values)) 
        for line in values:
            if np.isnan(line[1]): continue
            content.append(line)
        results[csv_file] = content
    print(results.keys())
    assert len(results) > 0
    sorted_keys = sorted(results.keys())
    sorted_values = [results[k][1:] for k in sorted_keys]
    # print(sorted_values.shape)
    content = [
        [query_x, query_y+'_mean', query_y+'_std']
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        # print(array)
        # assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        content.append(line)
    output_path = os.path.join(root_dir, query_y.replace('/', '_')+".csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)
    return output_path


def csv2numpy(file_path):
    df = pd.read_csv(file_path)
    step = df.iloc[:,0].to_numpy()
    mean = df.iloc[:,1].to_numpy()
    std = df.iloc[:,2].to_numpy()
    return step, mean, std


def smooth(y, radius=0):
    convkernel = np.ones(2 * radius + 1)
    out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
    return out


def plot_figure(
    results,
    x_label,
    y_label,
    xlim=None,
    ylim=None,
    title=None,
    smooth_radius=10,
    figsize=None,
    dpi=None,
    color_list=None,
    legend_outside=False
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if color_list == None:
        color_list = [COLORS[i] for i in range(len(results))]
    else:
        assert len(color_list) == len(results)
    for i, (algo_name, csv_file) in enumerate(results.items()):
        x, y, shaded = csv2numpy(csv_file)
        print(x.shape,y.shape, algo_name)
        y = smooth(y, smooth_radius)
        shaded = smooth(shaded, smooth_radius)
        if algo_name=='WBC':
            ax.plot(x, y, color=color_list[i], label=r'Ours-$\tau=0.7$,$\alpha$=0.2',linewidth=2.5)
        elif algo_name=='WBC-expect05':
            ax.plot(x,y,color=color_list[i], label=r'Ours-$\tau=0.5$',linewidth=2.5)
        elif algo_name=='WBC-expect09':
            ax.plot(x,y,color=color_list[i], label=r'Ours-$\tau=0.9$',linewidth=2.5)
        elif algo_name=='WBC-temper05':
            ax.plot(x,y,color=color_list[i], label=r'Ours-$\alpha=0.5$',linewidth=2.5)
        elif algo_name=='WBC-temper08':
            ax.plot(x,y,color=color_list[i], label=r'Ours-$\alpha=0.8$',linewidth=2.5)    
        else:
            ax.plot(x, y, color=color_list[i], label=algo_name,linewidth=2.5)
        ax.fill_between(x, y-shaded, y+shaded, color=color_list[i], alpha=0.15)
    ax.set_title(title, fontdict={'size': 15})
    ax.set_xlabel(x_label, fontdict={'size': 14})
    ax.set_ylabel(y_label, fontdict={'size': 14})
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend_outside:
        ax.legend(loc=2, bbox_to_anchor=(1,1), prop={'size': 12})
    else:
        ax.legend(prop={'size': 12})
    plt.legend()

def plot_func(
    root_dir,
    task,
    algos,
    query_file,
    query_x,
    query_y,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    title=None,
    smooth_radius=20,
    figsize=None,
    dpi=None,
    colors=None,
    legend_outside=True
):
    results = {}
    for algo in algos:
        path = os.path.join(root_dir, task, algo)
        csv_file = merge_csv(path, query_file, query_x, query_y)
        results[algo] = csv_file

    plt.style.use('seaborn')
    plot_figure(
        results=results,
        x_label=xlabel,
        y_label=ylabel,
        xlim=xlim,
        ylim=ylim,
        title=title,
        smooth_radius=smooth_radius,
        figsize=figsize,
        dpi=dpi,
        color_list=colors,
        legend_outside=legend_outside
    )
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotter")
    parser.add_argument("--root-dir", default="log")
    parser.add_argument("--task", default="hopper-medium-v2")
    # parser.add_argument("--algos", type=str, nargs='*', default=['td3bc'])
    parser.add_argument("--query-file", default="policy_training_progress.csv")
    parser.add_argument("--query-x", default="timestep")
    parser.add_argument("--query-y", default="eval/normalized_episode_reward")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="Timesteps")
    parser.add_argument("--ylabel", default='Normalize Score')
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--colors", type=str, nargs='*', default=None)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--output-path", default="./figures")
    parser.add_argument("--figsize", type=float, nargs=2, default=(8, 6))
    parser.add_argument('--save_name', type=str, default='baseline')
    parser.add_argument("--dpi", type=int, default=500)
    args = parser.parse_args()
    

    # algs = ['wtd3bc-temper02','td3bc','iql-reward_norm_temper3','edac','mcq','cql','awac']
    algs = ['WBC','AWAC','CQL','IQL','TD3BC','SQL','EDAC']
    # algs = ['WBC','IQL','SQL']
    # algs = ['WBC','wtd3bc-temper4']
    # algs = ['WBC-temper2']
    algs =['WBC','WBC-expect05','WBC-expect09','WBC-temper05','WBC-temper08']
    results = {}
    for algo in algs:
        path = os.path.join(args.root_dir, args.task, algo)
        csv_file = merge_csv(path, args.query_file, args.query_x, args.query_y)
        results[algo] = csv_file

    plt.style.use('seaborn')
    plot_figure(
        results=results,
        x_label=args.xlabel,
        y_label=args.ylabel,
        title=args.task,
        smooth_radius=args.smooth,
        figsize=args.figsize,
        dpi=args.dpi,
        color_list=args.colors,
        xlim=(0,1e6),
    )
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, args.task +'-'+ args.save_name +'.png')
        plt.savefig(save_path)
    if args.show:
        plt.show()