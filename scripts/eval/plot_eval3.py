"""Reproduce the detailed Eval_3 charts from assets/figures/eval3/results.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'assets/figures/eval3'
SNAPSHOT = DEST / 'results.json'
RENDER_DEST = DEST
COLORS = {'baseline': '#A3AAA7', 'sft100k': '#347B78', 'rlcd30k': '#C97558'}
MARKERS = {'baseline': 'o', 'sft100k': 's', 'rlcd30k': 'D'}
INK, MUTED, GRID, BG = '#203443', '#58656F', '#E9EEF1', '#FFFFFF'
METHODS = ('baseline', 'sft100k', 'rlcd30k')
DOMAIN_NAMES = {'vqa': 'Visual question answering', 'ui': 'User interfaces',
                'game': 'Games', 'document': 'Documents & charts'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def model_name(row):
    if row['method'] == 'baseline':
        return f"Qwen3.5-{row['size']}"
    suffix = 'SFT' if row['method'] == 'sft100k' else 'RLCD'
    return f"Valen-Base-{suffix}-{row['size']}"


PAPER, RULE, TEXT, SUBTLE = '#FFFFFF', '#DDDCD5', '#252B2B', '#727976'
TINTS = {'baseline': '#F2F3F0', 'sft100k': '#E7F1EE', 'rlcd30k': '#F8ECE6'}


def save(fig, name):
    options = dict(facecolor=PAPER)
    fig.savefig(RENDER_DEST / (name+'.png'), dpi=240, **options)
    fig.savefig(RENDER_DEST / (name+'.svg'), metadata={'Date': None}, **options)
    svg_path = RENDER_DEST / (name+'.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    fig.savefig(RENDER_DEST / (name+'.pdf'), metadata={'CreationDate': None, 'ModDate': None}, **options)
    plt.close(fig)


def canvas(height):
    fig = plt.figure(figsize=(10.8, height/100), facecolor=PAPER)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1080)
    ax.set_ylim(height,0)
    ax.set_axis_off()
    return fig, ax


def text(ax,x,y,label,size=11,color=TEXT,weight='normal',ha='left',**kw):
    return ax.text(x,y,label,fontsize=size,color=color,weight=weight,ha=ha,va='center',**kw)


def line(ax,x1,y1,x2,y2,color=RULE,lw=.65):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,solid_capstyle='butt',zorder=1)


def rect(ax,x,y,w,h,color):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='none',zorder=2))


def row_label(ax,row,y):
    rect(ax,24,y-7,3,14,COLORS[row['method']])
    text(ax,38,y,model_name(row),size=11.2,
         weight='bold' if row['method']!='baseline' else 'normal')


def table_header(ax, columns):
    for x,label,align in columns:
        text(ax,x,22,label,size=10,color=SUBTLE,ha=align)
    line(ax,24,44,1056,44,color='#999E99',lw=.8)


def overall(models, latency=False):
    rows=[r for r in models if not latency or r['method']!='baseline']
    step=43; top=70; bottom=top+(len(rows)-1)*step
    fig,ax=canvas(bottom+57)
    table_header(ax,[(38,'MODEL','left'),(363,'MEAN LATENCY  ·  ms ↓' if latency else 'ACCURACY  ·  % ↑','left'),
                     (903,'TIME' if latency else 'SCORE','right'),
                     (1040,'vs. baseline','right')])
    low,high=(0,200) if latency else (60,85)
    x0,x1=363,809
    ticks=[0,50,100,150,200] if latency else [60,65,70,75,80,85]
    for tick in ticks:
        x=x0+(tick-low)/(high-low)*(x1-x0)
        line(ax,x,53,x,bottom+17,color='#ECEDE8',lw=.5)
        text(ax,x,bottom+36,str(tick),size=8.5,color=SUBTLE,ha='center')
    baselines={r['size']:r for r in models if r['method']=='baseline'}
    for i,row in enumerate(rows):
        y=top+i*step
        if i and row['size']!=rows[i-1]['size']:
            line(ax,24,y-step/2,1056,y-step/2,color='#BEC3BD',lw=.8)
        row_label(ax,row,y)
        value=row['mean_latency_ms'] if latency else row['accuracy_pct']
        assert low<=value<=high
        width=(value-low)/(high-low)*(x1-x0)
        # Explicitly labeled 60–85% accuracy axis; latency bars begin at zero.
        rect(ax,x0,y-8,width,16,COLORS[row['method']])
        text(ax,903,y,f'{value:.2f}',size=14,weight='bold',ha='right')
        base=baselines[row['size']]
        if row['method']=='baseline':
            text(ax,1040,y,'—',color=SUBTLE,ha='right')
        else:
            delta=f"{base['mean_latency_ms']/value:.2f}× faster" if latency else f"+{value-base['accuracy_pct']:.2f} pp"
            text(ax,1040,y,delta,size=10.5,color=COLORS[row['method']],ha='right')
    save(fig,'latency' if latency else 'accuracy')


def task_latency(models):
    rows=[r for r in models if r['method']!='baseline']
    fig,ax=canvas(286)
    # Align all three miniature bar charts to the same model rows.
    text(ax,38,24,'MODEL',size=10,color=SUBTLE)
    settings=[('choice','Choice',200),('noul','Noul',140),('score','Score',600)]
    step=222
    for j,(key,title,high) in enumerate(settings):
        start=363+j*step; end=start+185
        text(ax,start,19,title,size=12,weight='bold')
        text(ax,end,20,'ms ↓',size=9,color=SUBTLE,ha='right')
        for tick in [0,high/2,high]:
            x=start+tick/high*185
            line(ax,x,53,x,244,color='#ECEDE8',lw=.5)
            text(ax,x,266,f'{tick:g}',size=8.5,color=SUBTLE,ha='center')
    line(ax,24,44,1056,44,color='#999E99',lw=.8)
    for i,row in enumerate(rows):
        y=71+i*51
        row_label(ax,row,y+4)
        if i==2:line(ax,24,y-25,1056,y-25,color='#BEC3BD',lw=.8)
        for j,(key,title,high) in enumerate(settings):
            start=363+j*step;end=start+185
            value=row['task_latency_ms'][key]
            text(ax,end,y-4,f'{value:.1f}',size=11,ha='right',weight='bold')
            rect(ax,start,y+9,value/high*185,8,COLORS[row['method']])
    save(fig,'latency-by-task')


def domain_accuracy(models):
    fig,ax=canvas(347)
    columns=[('vqa','Visual QA'),('ui','Interfaces'),('game','Games'),('document','Docs & charts')]
    x0,cell=348,177
    text(ax,38,29,'MODEL',size=10,color=SUBTLE)
    for j,(domain,title) in enumerate(columns):
        x=x0+j*cell
        text(ax,x+cell/2,20,title,size=11.5,weight='bold',ha='center')
        count=models[0]['domains'][domain]['questions']
        text(ax,x+cell/2,39,f'n = {count:,}',size=8.5,color=SUBTLE,ha='center')
    line(ax,24,56,1056,56,color='#999E99',lw=.8)
    for i,row in enumerate(models):
        y=79+i*40
        row_label(ax,row,y)
        for j,(domain,title) in enumerate(columns):
            x=x0+j*cell
            value=row['domains'][domain]['accuracy_pct']
            best=max(r['domains'][domain]['accuracy_pct'] for r in models if r['size']==row['size'])
            winner=value==best
            if winner:
                rect(ax,x+6,y-17,cell-12,34,TINTS[row['method']])
                rect(ax,x+6,y-17,2,34,COLORS[row['method']])
            text(ax,x+cell/2,y,f'{value:.2f}',size=13,
                 color=TEXT,weight='bold' if winner else 'normal',ha='center')
        if i<5:
            line(ax,24,y+20,1056,y+20,color='#BEC3BD' if i==2 else '#EEEFEA',lw=.8 if i==2 else .5)
    line(ax,24,303,1056,303,color='#999E99',lw=.8)
    text(ax,38,325,'Accuracy (%) ↑',size=9,color=SUBTLE)
    text(ax,1040,325,'Shaded cell = best within each model size',size=9,color=SUBTLE,ha='right')
    save(fig,'accuracy-by-domain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEST, help='Render a preview to a separate directory')
    args = parser.parse_args()
    global RENDER_DEST
    RENDER_DEST = args.output_dir
    RENDER_DEST.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    family = 'Lato' if any(f.name == 'Lato' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 11, 'text.color': INK,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'valen-eval3',
                         'pdf.fonttype': 42,
                         'axes.unicode_minus': False})
    models = read(SNAPSHOT)['models']
    assert [(r['size'],r['method']) for r in models] == [(s,m) for s in ('0.8B','2B') for m in METHODS]
    overall(models)
    overall(models, latency=True)
    task_latency(models)
    domain_accuracy(models)
    print(f'Wrote four PNG/SVG/PDF charts to {RENDER_DEST}')


if __name__ == '__main__':
    main()
