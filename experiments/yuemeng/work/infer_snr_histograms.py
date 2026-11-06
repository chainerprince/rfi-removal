from pathlib import Path
import json
import sys
import time
import argparse
import os

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path,default=Path(os.environ.get('RFI_BASELINE_DIR', str(ROOT/'inputs/baseline'))))
parser.add_argument('--data-dir',type=Path,default=Path(os.environ.get('RFI_DATA_DIR', str(ROOT/'data/starter-v1.0'))))
parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/baseline_histograms')
args=parser.parse_args()
RUN = args.run_dir
DATA = args.data_dir
CACHE = args.output_dir
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DATA.parents[1] / 'src'))
sys.path.insert(0, str(ROOT))
from reference import training_core as core

calibration = json.loads((RUN/'calibration.json').read_text())
config = json.loads((RUN/'config.json').read_text())
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
checkpoint=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True)
assert core.settings_match(checkpoint['config'], core.training_settings(config))
model = core.SegmentationModel(base=config['base_channels'], use_skips=config['architecture'] == 'unet').eval()
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
catalog = core.load_catalog(DATA)
device = torch.device('cpu')
validation=catalog.query("split == 'validation'").sort_values('sample_id')
bins = config['probability_bins']
cut = calibration['quantized_cut']
assert calibration['probability_bins'] == bins
assert calibration.get('selected_epoch', checkpoint['epoch']) == checkpoint['epoch']
clean_ids = validation.loc[validation.rfi_type.eq('clean'), 'sample_id'].tolist()
assert clean_ids == sorted(calibration['clean_sample_ids'])
ph_all=[]; nh_all=[]; count_rows=[]
started=time.perf_counter()
with torch.inference_mode():
    for row in validation.itertuples():
        rec=core.load_recording(DATA/row.path)
        ph=np.zeros(bins,np.int64);nh=np.zeros(bins,np.int64)
        for antenna in range(10):
            raw=rec['power'][:,antenna,:,0]
            truth=rec['rfi_mask'][:,antenna,:,0]
            prob=core.predict_probability(model,raw,checkpoint['normalization'],device)
            a,b=core.probability_histograms(prob,truth,bins)
            ph+=a;nh+=b
            tp=int(a[cut:].sum());fp=int(b[cut:].sum())
            count_rows.append(dict(sample_id=int(row.sample_id),antenna=antenna,tp=tp,fp=fp,
                                   fn=int(a.sum())-tp,tn=int(b.sum())-fp))
        ph_all.append(ph);nh_all.append(nh)
        if len(ph_all)%10==0:
            print(f'INFERENCE_ONLY {len(ph_all)}/{len(validation)} recordings; {time.perf_counter()-started:.1f}s',flush=True)
np.savez_compressed(CACHE/'per_recording_histograms_cpu.npz',
                    sample_id=validation.sample_id.to_numpy(),positive=np.asarray(ph_all),negative=np.asarray(nh_all),bins=bins)
counts=pd.DataFrame(count_rows)
counts.to_csv(CACHE/'cpu_threshold_counts.csv',index=False)
original=pd.read_csv(next(RUN.glob('evaluation_validation_*/per_antenna.csv'), RUN/'per_antenna.csv'))
original=original.query("method == 'model_matched_clean_fpr'")
comparison=original[['sample_id','antenna','tp','fp','fn','tn']].merge(counts,on=['sample_id','antenna'],suffixes=('_saved','_cpu'),validate='one_to_one')
for key in ['tp','fp','fn','tn']:
    comparison[key+'_delta']=comparison[key+'_cpu']-comparison[key+'_saved']
comparison.to_csv(CACHE/'saved_vs_cpu_counts.csv',index=False)
total_ap=core.precision_recall_from_histograms(np.sum(ph_all,axis=0),np.sum(nh_all,axis=0))['ap']
audit=dict(device='cpu',torch_version=torch.__version__,numpy_version=np.__version__,
           selected_epoch=checkpoint['epoch'],inference_only=True,
           optimizer_steps=0,normalization=checkpoint['normalization'],bins=bins,cut=cut,
           recordings=len(validation),antenna_pairs=len(counts),seconds=time.perf_counter()-started,
           original_ap=checkpoint['validation_ap'],cpu_ap=total_ap,ap_delta=total_ap-checkpoint['validation_ap'],
           absolute_count_deltas={key:int(comparison[key+'_delta'].abs().sum()) for key in ['tp','fp','fn','tn']},
           primary_confusion_metrics_source='Original saved CUDA per_antenna.csv; CPU inference supplies SNR-specific AP only.')
(CACHE/'inference_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
