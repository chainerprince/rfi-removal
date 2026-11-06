from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--catalog',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--histograms',type=Path)
parser.add_argument('--bootstrap',type=int,default=2000)
args=parser.parse_args()
out=args.output_dir
(out / 'tables').mkdir(parents=True, exist_ok=True)
eval_dir=next(args.run_dir.glob('evaluation_validation_*'), args.run_dir)
summary = json.loads((eval_dir / 'summary.json').read_text())
catalog = pd.read_csv(args.catalog)
saved = pd.read_csv(eval_dir / 'per_antenna.csv')
COUNTS = ['tp', 'fp', 'fn', 'tn']
validation_ids = set(catalog.loc[catalog.split.eq('validation'), 'sample_id'])
if set(saved.sample_id) != validation_ids or not saved['split'].eq('validation').all():
    raise ValueError('Expected the original validation predictions')
if saved.duplicated(['sample_id', 'antenna', 'method']).any():
    raise ValueError('Duplicate per-antenna evaluation results')
coverage = saved.groupby(['method', 'sample_id']).size()
if len(coverage) != saved.method.nunique() * len(validation_ids) or not coverage.eq(10).all():
    raise ValueError('Expected ten antennas per method and recording')

edges=np.array([.01,.1,1.,10.,np.nextafter(100.,np.inf)])
labels=['[0.01, 0.1)','[0.1, 1)','[1, 10)','[10, 100]']
lookup=catalog.drop(columns=['split','rfi_type'])
joined=saved.merge(lookup,on='sample_id',validate='many_to_one')
joined['reference_inr_db']=np.nan
mask=joined.reference_power_ratio.gt(0)
joined.loc[mask,'reference_inr_db']=10*np.log10(joined.loc[mask,'reference_power_ratio'])
joined['snr_bin']=pd.cut(joined.reference_power_ratio,edges,right=False,labels=labels).astype('object')
joined.loc[~mask,'snr_bin']='control'
assert joined.snr_bin.notna().all()
joined.to_csv(out/'tables/per_antenna_with_snr.csv',index=False)

def rates(c):
    c=np.asarray(c,dtype=float)
    tp,fp,fn,tn=np.moveaxis(c,-1,0)
    def ratio(a,b):
        return np.divide(a,b,out=np.full(np.shape(a),np.nan),where=b>0)
    return dict(recall=ratio(tp,tp+fn),precision=ratio(tp,tp+fp),fpr=ratio(fp,fp+tn),
                positive_fraction=ratio(tp+fn,tp+fp+fn+tn))

record=joined.groupby(['sample_id','method'],sort=True)[COUNTS].sum().reset_index()
record=record.merge(joined[['sample_id','rfi_type','reference_power_ratio','reference_inr_db','snr_bin','duty_cycle','mean_power_ratio']].drop_duplicates('sample_id'),on='sample_id',validate='many_to_one')
for k,v in rates(record[COUNTS].to_numpy()).items():record[k]=v
record.to_csv(out/'tables/per_recording.csv',index=False)
rng=np.random.default_rng(20261026)
rows=[];draws={}
for method in sorted(record.method.unique()):
    for label in labels+['control']:
        g=record.loc[record.method.eq(method)&record.snr_bin.eq(label)]
        if g.empty: continue
        c=g[COUNTS].to_numpy(np.int64);values=rates(c.sum(axis=0))
        total=c.sum(axis=0)
        boot=c[rng.integers(0,len(c),size=(args.bootstrap,len(c)))].sum(axis=1)
        br=rates(boot)
        row=dict(method=method,snr_bin=label,recordings=len(g),antenna_pairs=10*len(g),
                 recordings_with_positive_labels=int((g.tp+g.fn).gt(0).sum()),
                 **dict(zip(COUNTS,map(int,total))),**{k:float(v) for k,v in values.items()},
                 macro_recall_mean=float(g.recall.mean()),macro_recall_sd=float(g.recall.std(ddof=1)))
        for key in ['recall','precision','fpr']:
            b=br[key];finite=b[np.isfinite(b)]
            row[key+'_ci_low'],row[key+'_ci_high']=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
        draws[(method,label)]=br['recall']
        rows.append(row)
grouped=pd.DataFrame(rows)
ap_by_bin={};replay_ap=np.nan
if args.histograms:
    cache=np.load(args.histograms)
    ids=cache['sample_id'];ph=cache['positive'];nh=cache['negative'];bins=int(cache['bins'])
    if len(ids) != len(validation_ids) or set(ids) != validation_ids:
        raise ValueError('The score cache does not match the validation split')
    cat=catalog.set_index('sample_id').loc[ids]
    def ap(a,b):
        if a.sum()==0:return np.nan
        present=np.r_[a,b]>0
        return float(average_precision_score(np.r_[np.ones(bins),np.zeros(bins)][present],np.tile(np.arange(bins),2)[present],sample_weight=np.r_[a,b][present]))
    replay_ap=ap(ph.sum(axis=0),nh.sum(axis=0))
    assert abs(replay_ap-summary['pixel_ap_quantized'])<1e-6
    strength=np.full(len(cat),np.nan)
    has=cat.reference_power_ratio.to_numpy()>0
    strength[has]=cat.reference_power_ratio.to_numpy()[has]
    group=pd.cut(strength,edges,right=False,labels=labels).astype('object')
    for label in labels:
        where=group==label
        ap_by_bin[label]=ap(ph[where].sum(axis=0),nh[where].sum(axis=0))
    observed=record.query("method == 'model_matched_clean_fpr'").set_index('sample_id').loc[ids]
    assert np.array_equal(ph.sum(axis=1),observed.tp.to_numpy()+observed.fn.to_numpy())
    assert np.array_equal(nh.sum(axis=1),observed.fp.to_numpy()+observed.tn.to_numpy())
grouped['pixel_ap_cpu_replay']=np.nan
for k,v in ap_by_bin.items():grouped.loc[grouped.method.eq('model_matched_clean_fpr')&grouped.snr_bin.eq(k),'pixel_ap_cpu_replay']=v
grouped.to_csv(out/'tables/metrics_by_snr_all_methods.csv',index=False)
model=grouped.query("method == 'model_matched_clean_fpr'").set_index('snr_bin').loc[labels].reset_index()
model.to_csv(out/'tables/model_metrics_by_snr.csv',index=False)
control_rows=[]
for (method,family),g in record.query("snr_bin == 'control'").groupby(['method','rfi_type']):
    c=g[COUNTS].sum().to_numpy()
    control_rows.append(dict(method=method,rfi_type=family,recordings=len(g),**dict(zip(COUNTS,map(int,c))),**{k:float(v) for k,v in rates(c).items()}))
controls=pd.DataFrame(control_rows)
controls.to_csv(out/'tables/controls.csv',index=False)

primary=record.query("method == 'model_matched_clean_fpr' and snr_bin != 'control'")
coverage=[]
for (family,label),g in primary.groupby(['rfi_type','snr_bin']):
    c=g[COUNTS].sum().to_numpy()
    coverage.append(dict(rfi_type=family,snr_bin=label,recordings=len(g),**dict(zip(COUNTS,map(int,c))),**{k:float(v) for k,v in rates(c).items()}))
coverage=pd.DataFrame(coverage)
coverage.to_csv(out/'tables/family_snr_coverage.csv',index=False)
delta=draws[('model_matched_clean_fpr',labels[1])]-draws[('model_matched_clean_fpr',labels[0])]
contrast=dict(description='Pooled recall difference: [0.1,1) minus [0.01,0.1) power ratio; descriptive, confounded by family/duty cycle',
              difference=float(model.iloc[1].recall-model.iloc[0].recall),ci_low=float(np.nanquantile(delta,.025)),ci_high=float(np.nanquantile(delta,.975)))

audit = dict(validation_recordings=len(validation_ids), rfi_recordings=int(primary.sample_id.nunique()),
             bootstrap_replicates=args.bootstrap, bootstrap_seed=20261026,
             original_ap=summary['pixel_ap_quantized'], cpu_replay_ap=float(replay_ap),
             bin_recordings=model.recordings.tolist(), contrast=contrast, no_training=True)
(out/'audit.json').write_text(json.dumps(audit,indent=2)+"\n")
print(model[['snr_bin','recordings','recall','precision','pixel_ap_cpu_replay']].to_string(index=False))
print('OUTPUT', out.resolve())
