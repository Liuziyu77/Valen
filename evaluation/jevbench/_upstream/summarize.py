# Vendored unchanged from fstandhartinger/jevbench, revision
# bb05a335bc809e61b20c0f745d25499a82b326fc. See LICENSE in this directory.
"""Allowlisted aggregates only; private item IDs/text/labels never exported."""
from collections import defaultdict
from .scoring import argmax_label
from .metrics import ece_top_label,latency_summary

def metric(tasks,records):
 by={t.id:t for t in tasks};rs=[r for r in records if r['task_id']in by];valid=[r for r in rs if r.get('valid')];scorable=[r for r in rs if by[r['task_id']].expected is not None and not by[r['task_id']].provenance.get('exclude_reason')]
 probs=[r for r in scorable if r.get('probs')];bs=[];pairs=[];mae=[]
 for r in probs:
  t=by[r['task_id']];p=r['probs'];gold=str(t.expected)
  bs.append(sum((p[k]-int(k==gold))**2 for k in t.labels));pairs.append((max(p.values()),argmax_label(p)==gold))
  if t.question['type']=='score':mae.append(abs(sum(float(k)*v for k,v in p.items())-t.expected))
 cost=[r.get('cost_usd')for r in rs];lat=[r['latency_s']for r in rs if r.get('latency_s')is not None]
 return {'n_planned':len(tasks),'n_attempted':len(rs),'n_scorable':len(scorable),'n_valid':len(valid),'n_correct':sum(bool(r.get('correct'))for r in scorable),'accuracy':sum(bool(r.get('correct'))for r in scorable)/len(scorable)if scorable else None,'coverage':len(rs)/len(tasks)if tasks else None,'schema_validity':len(valid)/len(rs)if rs else None,'schema_validity_strict':sum(1 for r in rs if r.get('strict_valid'))/len(rs)if rs else None,'n_renormalized':sum(1 for r in rs if r.get('renormalized')),'operational_success':sum(r['ok']for r in rs)/len(rs)if rs else None,'calibration_n':len(probs),'brier_mean':sum(bs)/len(bs)if bs else None,'ece':ece_top_label(pairs)if pairs else None,'ordinal_mae':sum(mae)/len(mae)if mae else None,'latency':latency_summary(lat),'latency_failures':latency_summary([r['latency_s']for r in rs if not r['ok']]),'price_per_1000_decisions_usd':sum(cost)*1000/len(rs)if rs and all(c is not None for c in cost)else None,'cost_basis':sorted({r.get('cost_basis','unknown')for r in rs})}
def agreement(tasks,records):
 by={r['task_id']:r for r in records};groups=defaultdict(list)
 for t in tasks:
  if t.group:groups[t.group].append(t)
 pairs=[g for g in groups.values()if len(g)==2];valid=[]
 for a,b in pairs:
  x,y=by.get(a.id),by.get(b.id)
  if x and y and x.get('valid')and y.get('valid'):valid.append((x,y))
 return {'pairs':len(pairs),'both_valid':len(valid),'agree':sum(a['predicted']==b['predicted']for a,b in valid),'agreement':sum(a['predicted']==b['predicted']for a,b in valid)/len(valid)if valid else None,'both_correct_rate_all_pairs':sum(bool(a['correct'])and bool(b['correct'])for a,b in valid)/len(pairs)if pairs else None}
def summarize(tasks,records,ledger_charged=None,headline_only=True):
 # ledger_charged is the shared job ledger's total, not this run's tariff cost;
 # it is reported beside the per-decision price, never folded into it.
 if len({r['task_id']for r in records})!=len(records):raise ValueError('Duplicate per-item records')
 if not {r['task_id']for r in records}<={t.id for t in tasks}:raise ValueError('Unknown task in run')
 fam={f:metric([t for t in tasks if t.family==f],records)for f in sorted({t.family for t in tasks})}
 acc=[v['accuracy']for v in fam.values()if v['accuracy']is not None];out=metric(tasks,records)
 out.update(version='jevbench-v1',macro_accuracy=sum(acc)/len(acc)if acc else None,per_family=fam,complete=len(records)==len(tasks),paraphrase_consistency=agreement(tasks,records),model_identities=sorted({r.get('model','unknown')for r in records}),probability_sources=sorted({r.get('probs_source','unknown')for r in records}))
 out['splits']={s:metric([t for t in tasks if t.split==s],records)for s in sorted({t.split for t in tasks})}
 out['ledger_charged_usd']=ledger_charged
 return out

def public_export(summary,tasks,records):
 # Recompute from trusted metric fields, never copy arbitrary caller-supplied fields.
 # Job spend is a private operational number and is recomputed as absent here.
 return summarize(tasks,records)
