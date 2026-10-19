"""Build briefing figures from committed aggregate results; no GPU or raw data.

Run: MPLCONFIGDIR=/tmp/umrm-mpl python tools/build_briefing_assets.py
Requires matplotlib and numpy. Outputs PNG/SVG plus the exact plotted numbers.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/2026-10-19/figures'
EGO = ROOT / 'docs/results/ego_pretrain_20261019/metrics.json'
HEAD = ROOT / 'docs/results/lastlayer/gates_ll.json'
COLORS = ['#64748b', '#2563eb', '#10a37f', '#e69b28']


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=190, bbox_inches='tight')
    svg = OUT / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight')
    # Matplotlib emits trailing spaces in SVG paths; normalize for Git checks.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def forest(ax, rows, title, xlabel):
    for i, (_, d) in enumerate(rows):
        v = 100 * d['diff']
        lo, hi = np.array(d['ci95']) * 100
        ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt='o', capsize=4,
                    color='#10a37f' if lo > 0 else '#b45309' if hi < 0 else '#475569')
        ax.annotate(f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]', (v, i),
                    xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows)-0.4, -0.65)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.18)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = json.loads(EGO.read_text())
    h = json.loads(HEAD.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    source = {'sources': [str(EGO.relative_to(ROOT)), str(HEAD.relative_to(ROOT))]}

    arms = ['A4', 'A36', 'S36', 'V36']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios':[1, 1.7]})
    vals = [h['arms'][a]['R@10'] for a in arms]
    axes[0].bar(arms, vals, color=COLORS)
    for i, v in enumerate(vals):
        axes[0].text(i, v+.006, f'{v:.4f}', ha='center')
    axes[0].set(ylim=(0,.36), ylabel='TMR R@10', title='Same SFT text, different action interfaces')
    rows = [(label, h['decomposition'][key]['R@10']) for label,key in [
        ('A36 - A4', 'D1_layer(A36-A4)'),
        ('S36 - A36', 'D2_sft_weights(S36-A36)'),
        ('V36 - S36', 'D3_video_context(V36-S36)')]]
    forest(axes[1], rows, 'Paired differences, source-cluster 95% CI', 'R@10 difference (percentage points)')
    fig.suptitle('Action interface | VAL 125, 34 sources, 3 motion draws/video, one training seed', fontsize=12, y=1.04)
    fig.tight_layout(); save(fig, '01_action_interface')
    source['action_interface'] = {'arms': h['arms'], 'comparisons': dict(rows)}

    keys = ['egonormia_verified','egotom_actions','egotom_belief','egotom_goal']
    names = ['EgoNormia\nn=200','EgoToM actions\nn=354','EgoToM belief\nn=332','EgoToM goal\nn=349']
    models = [('base','Base 4B'),('sft','Original SFT'),('pt_c','Ego PT-C'),('sft_ptc','PT-C + original SFT')]
    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(4)
    for i,(key,label) in enumerate(models):
        vals = [e['probe_acc'][key][k]['acc']*100 for k in keys]
        bars = ax.bar(x+(i-1.5)*.19, vals, width=.18, label=label, color=COLORS[i])
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    for i,k in enumerate(keys):
        chance=e['probe_acc']['base'][k]['chance']*100
        ax.plot([i-.42,i+.42],[chance,chance], color='black',ls=':',lw=1)
    ax.set_xticks(x,names); ax.set(ylabel='Option-scoring accuracy (%)',ylim=(0,80))
    ax.set_title('Ego probes: skill acquisition and retention through original SFT')
    ax.legend(loc='upper right',ncol=2,frameon=False)
    ax.text(.01,-.2,'Dotted lines: chance (25%, 25%, 25%, 33.3%). Bars are point estimates; paired CIs are in Figure 3.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout(); save(fig,'02_ego_probes')
    source['probe_accuracy'] = e['probe_acc']

    fig, axes = plt.subplots(1,3,figsize=(17,5),sharey=True)
    labels=['EgoNormia','Actions','Belief','Goal','EgoToM pooled']
    compare=[('pt_c_vs_base','Acquisition: PT-C - base'),
             ('sft_ptc_vs_pt_c','Retention: final - PT-C'),
             ('sft_ptc_vs_sft','Final difference vs original SFT')]
    for ax,(key,title) in zip(axes,compare):
        rows=[(n,e['probe_comparisons'][key][k]) for n,k in zip(labels,keys+['egotom_pooled'])]
        forest(ax,rows,title,'Accuracy difference (percentage points)')
        ax.set_xlim(-15,24)
    fig.suptitle('Paired video-UID cluster bootstrap 95% CIs | secondary comparisons exploratory',y=1.02)
    fig.tight_layout(); save(fig,'03_retention_differences')
    source['probe_comparisons'] = {k:e['probe_comparisons'][k] for k,_ in compare}

    fig, axes = plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[1,1.25]})
    for key,label,color in [('sft_student','Base initialization',COLORS[1]),('sft_ptc','Ego initialization',COLORS[2])]:
        vals=e['sft_validation'][key]
        axes[0].plot([x['epoch'] for x in vals],[x['val_nll_per_tok'] for x in vals],'-o',label=label,color=color)
    axes[0].set(xlabel='UMRM SFT epoch',ylabel='VAL NLL / token',xticks=[0,1,2],title='Better starting NLL; similar after SFT')
    axes[0].legend(frameon=False)
    axes[0].annotate('Selected epoch 1',xy=(1,.70),xytext=(.7,1.12),arrowprops={'arrowstyle':'->','color':'#64748b'})
    forest(axes[1],[('PT-C + SFT\nminus original SFT',e['motion']['R@10'])],
           'G-E1: direct motion transfer (no OPD)', 'A36 R@10 difference (percentage points)')
    axes[1].set_xlim(-8,11)
    axes[1].text(.5,.10,'0.3040 vs 0.2907\n95% CI crosses zero; G-E1 did not pass',ha='center',transform=axes[1].transAxes)
    fig.tight_layout(); save(fig,'04_sft_and_motion')
    source['sft_validation']=e['sft_validation']; source['motion_R10']=e['motion']['R@10']
    (OUT/'plotted_values.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print(f'Wrote 4 PNG, 4 SVG and plotted_values.json to {OUT}')


if __name__ == '__main__':
    main()
