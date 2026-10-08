import os, sys, json, subprocess, time, traceback, hashlib
from pathlib import Path
import torch

SH=int(sys.argv[1]); W=Path(sys.argv[2]); CODE=W/'code'
D=Path('/data/caiqingran/enzymecot'); S=D/'cot_split_v2'
PY='/data/caiqingran/env/py/bin/python'
CK=D/'runs/q35_deep2like_4stage_16g/FINAL_d2like_step9000_val0.7807.pt'
Q=D/'hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
base=W/'shards'/f'shard{SH}';base.mkdir(parents=True,exist_ok=True)
status_path=base/'pipeline_status.json'
def status(phase,**kw):
 v={'phase':phase,'shard':SH,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
 t=status_path.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(status_path)
def read_fasta(p):
 d={};k=None
 for l in p.read_text().splitlines():
  if l.startswith('>'):
   k=l[1:].split('|')[0].split()[0]
   if k in d:raise ValueError(f'duplicate fasta ID {k}')
   d[k]=''
  elif k:d[k]+=l.strip()
 return d

def main():
 rows=json.load(open(S/'test_rows_cotfull.json'))[SH::16]
 want={str(r['idx']) for r in rows}
 docs=torch.load(W/f'testdocs_docs_{SH}.pt',map_location='cpu',weights_only=False)
 raw=torch.load(W/f'testdocs_pairs_{SH}.pt',map_location='cpu',weights_only=False)
 assert set(map(str,docs['ids']))==want, 'writer did not cover exact shard'
 assert set(map(str,docs['spans']))==want and set(map(str,docs['hard']))==want
 rawpairs={str(k):v for k,v in raw['pairs'].items()}
 pairs={};pos={};invalid={};stats=[]
 for r in rows:
  k=str(r['idx']);L=int(r['length']);pairs[k]={};invalid[k]=[]
  for st,pp in rawpairs.get(k,{}).items():
   kept=[]
   for aa,p in pp:
    if aa in 'ACDEFGHIKLMNPQRSTVWY' and 0<=int(p)<L:kept.append((aa,int(p)))
    else:invalid[k].append([int(st),aa,int(p)])
   pairs[k][int(st)]=kept
  core={p for aa,p in pairs[k].get(2,[])}
  site={p for aa,p in pairs[k].get(3,[])}
  # Preserve the checkpoint's legacy 0-based/token-grid convention. Do not
  # silently fix its BOS offset only at inference. Raw model outputs are kept.
  pos[k]={'core':sorted(core),'pocket':sorted(site-core)}
  stats.append({'idx':k,'length':L,'pairs_by_stage':{str(t):len(pairs[k].get(t,[])) for t in (2,3,4,5)},'invalid_pairs':len(invalid[k])})
 torch.save({'pairs':pairs,'aa_alphabet':'ACDEFGHIKLMNPQRSTVWY','source':'new checkpoint generated stage text; out-of-length predictions filtered; empty outputs retained'},base/'pairs.pt')
 json.dump(rows,open(base/'rows.json','w'));json.dump(pos,open(base/'predicted_positions.json','w'))
 json.dump({'n':len(rows),'stats':stats,'invalid':invalid,'empty_pair_records':sum(not any(v.values()) for v in pairs.values())},open(base/'writer_audit.json','w'),indent=2)
 reference=json.load(open(S/'core_pos.json'))
 assert want<=set(reference), 'reference-position comparison missing IDs'
 json.dump({k:reference[k] for k in want},open(base/'reference_positions.json','w'))
 env=os.environ.copy();env.update(MR_TRIM_PAD='1',MR_EVAL_SKIP_TEST='1',MR_TEXT_DROPOUT='0',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 clean_failures=[]
 for arm in ('predicted_positions','reference_positions'):
  out=base/arm;out.mkdir(exist_ok=True)
  for sp in ('train','val','test'):
   for kind in ('dplm2_tokens','no_retrieval_tokens'):
    dst=out/f'{sp}_{kind}.pt';src=D/'tokcache'/dst.name
    # Each arm owns its files: the seed cache covers only part of the test set.
    # Sharing a writable symlink here would let workers overwrite one another.
    previous=base/'predicted_positions'/dst.name
    if arm=='reference_positions' and previous.is_file():src=previous
    assert src.is_file(),src
    if dst.is_symlink():dst.unlink()
    if not dst.exists():
     cached=torch.load(src,map_location='cpu',weights_only=False)
     if sp=='test':cached=[r for r in cached if str(r['idx']) in want]
     torch.save(cached,dst)
  done=out/'generation_verified.json'
  if not done.exists():
   cmd=[PY,'-u',str(CODE/'dplm2bit_textcond_hybrid.py'),
    '--train_rows',str(S/'train_rows_cotfull.json'),'--val_rows',str(S/'val_rows_cotfull.json'),
    '--test_rows',str(base/'rows.json'),'--split_json',str(S/'split_seqcluster.json'),
    '--out_dir',str(out),'--model_name','/data/caiqingran/models/dplm2_bit_650m',
    '--struct_tokenizer_name',str(D/'dplm2_struct_tokenizer_hydra_wrapper'),'--init_checkpoint',str(CK),
    '--per_token_text','--max_text_tokens','576','--qwen_path',str(Q),
    '--text_embed_pt',str(S/'text_embed_hash.pt'),'--qwen_docs',str(W/f'testdocs_docs_{SH}.pt'),
    '--qwen_hidden_mean',str(S/'qwen35_stage_means_v2.pt'),
    '--nce_gamma','-1','--nce_ratio','0.1','--nce_bank',str(S/'text_bank_stage1_q35_len997.pt'),
    '--text_lambda','0.7277','--qwen_lora_r','32','--site_pairs',str(base/'pairs.pt'),'--ground_mode','gated',
    '--core_pos_json',str(base/f'{arm}.json'),'--resid_scale','0','--ctx_corrupt','0.15',
    '--max_len','9999','--steps','0','--batch','1','--unfreeze_last','8',
    '--train_limit','8','--train_eval_limit','8','--val_limit','8','--test_limit','100000',
    '--eval_interval','100000','--loss_log_interval','100000','--skip_step1_eval','--eval_batch_size','4',
    '--tf32','--gen_bf16','--generate_only_test','--multiround_generate','--generate_batch_size','4',
    '--sample_max_iter','500','--sample_strategy','annealing@1.1:0.1','--no_retrieval',
    '--use_mmdit_text','--mmdit_layers','1','--mmdit_text_tokens','4','--objective','multiround',
    '--fullmask_gamma','1.0','--high_noise_gamma','0.5','--struct_loss_gamma','1.0','--aa_loss_gamma','1.0',
    '--diffusion_weighting','constant','--use_deep_per_token_cross','--deep_cross_layers','8','--seed','10320']
   json.dump({'command':cmd,'checkpoint':str(CK),'arm':arm,'ids':sorted(want),'legacy_indexing':True,'dynamic_structure_feedback':False},open(out/'execution.json','w'),indent=2)
   status('generating',arm=arm,n=len(rows))
   with open(out/'generation.log','a') as log:
    result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    raise RuntimeError(f'generation {arm} exit={result.returncode}, see {out}/generation.log')
   fasta=out/'generated_test/aatype.fasta';seqs=read_fasta(fasta)
   assert set(seqs)==want, f'generated ID mismatch missing={want-set(seqs)} extra={set(seqs)-want}'
   mismatches=[{'idx':str(r['idx']),'actual':len(seqs[str(r['idx'])]),'expected':r['length']} for r in rows if len(seqs[str(r['idx'])])!=int(r['length'])]
   if mismatches:raise RuntimeError(f'generated length mismatch: {mismatches[:5]}')
   bad={k:sorted(set(v)-set('ACDEFGHIKLMNPQRSTVWY')) for k,v in seqs.items() if set(v)-set('ACDEFGHIKLMNPQRSTVWY')}
   if bad:raise RuntimeError(f'invalid sequence alphabet: {bad}')
   json.dump({'n':len(seqs),'ids':sorted(seqs),'native_identical':sum(seqs[str(r['idx'])]==r['sequence'] for r in rows),'fasta_sha256':hashlib.sha256(fasta.read_bytes()).hexdigest(),'arm':arm},open(done,'w'),indent=2)
  status('clean',arm=arm)
  clean=out/'clean';clean.mkdir(exist_ok=True)
  cmd=[PY,str(CODE/'score_clean_shard.py'),'--fasta',str(out/'generated_test/aatype.fasta'),'--rows',str(base/'rows.json'),'--output',str(clean),'--name-prefix',f'd2s9000_20261002_{arm}_s{SH}']
  with open(out/'clean.log','a') as log:
   scored=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  if scored.returncode:clean_failures.append({'arm':arm,'returncode':scored.returncode,'log':str(out/'clean.log')})
 if clean_failures:raise RuntimeError(f'Generation finished but CLEAN incomplete: {clean_failures}')
 status('complete',n=len(rows),arms=['predicted_positions','reference_positions'])

try:main()
except Exception:
 status('failed',error=traceback.format_exc());raise
