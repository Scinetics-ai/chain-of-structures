import os, sys, json, time, subprocess, traceback
from pathlib import Path
shard=int(sys.argv[1]); root=Path(sys.argv[2]); code=root/'code'
py='/data/caiqingran/env/py/bin/python'; split='/data/caiqingran/enzymecot/cot_split_v2'
ck='/data/caiqingran/enzymecot/runs/q35_deep2like_4stage_16g/FINAL_d2like_step9000_val0.7807.pt'
q='/data/caiqingran/enzymecot/hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
root.joinpath('status').mkdir(exist_ok=True); root.joinpath('logs').mkdir(exist_ok=True)
state={'shard':shard,'pid':os.getpid(),'host':os.uname().nodename,'checkpoint':ck,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def status(phase,**kw):
 state.update(phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 p=root/'status'/f'shard{shard}.json'; t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
env=os.environ.copy();env.update(SPLIT_DIR=split,DOCS_PT=split+'/qwen35_9b_docs_v2.pt',QWEN_PATH=q,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONPATH='/data/caiqingran/code/dplm:/data/caiqingran/code/dplm/src:/data/caiqingran/code/dplm/vendor/openfold')
try:
 status('writer')
 cmd=[py,'-u',str(code/'gen_cot_docs_v2.py'),'--lora',ck,'--rows',split+'/test_rows_cotfull.json','--shard',str(shard),'--nshard','16','--batch','8','--max_new','2000','--max_pos','9999','--out',str(root/'testdocs')]
 with open(root/'logs'/f'writer_{shard}.log','a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('writer_done')
 ready=code/'generation_ready.json'
 deadline=time.time()+7200
 while not ready.exists():
  if time.time()>deadline:raise RuntimeError('generation configuration not installed within 2h')
  time.sleep(10)
 status('generation_pipeline')
 with open(root/'logs'/f'pipeline_{shard}.log','a') as log:subprocess.run([py,'-u',str(code/'continue_eval.py'),str(shard),str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('complete')
except Exception:
 status('failed',error=traceback.format_exc());raise
