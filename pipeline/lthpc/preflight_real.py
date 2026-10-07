"""Real local-weight acceptance probes; never claim biological efficacy."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component', choices=['policy', 'native', 'joint'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source')
    args = p.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    checks = {}
    started = time.time()
    def emit(event, **items):
        print(json.dumps({'event':event, **items}, allow_nan=False), flush=True)
    def check(name, result, **items):
        checks[name] = {'passed': bool(result), **items}
        emit('check', name=name, **checks[name])
        if not result:
            raise AssertionError(name)
    receipt = {'component':args.component, 'scope':'real_weight_engineering_preflight_not_quality',
               'config':config, 'pid':os.getpid(), 'checks':checks, 'passed':False}
    try:
        torch.manual_seed(config['seed'])
        device = config['device']
        q, g = config['qwen'], config['dplm']
        if args.component == 'policy':
            from transformers import AutoTokenizer
            from agent_dplmbit.policy import load_qwen_policy, prepare_messages
            tok = AutoTokenizer.from_pretrained(q['model_path'],local_files_only=True)
            model = load_qwen_policy(q['model_path'], num_condition_tokens=q['num_condition_tokens'],
                                     lora_rank=q['lora_rank'], lora_alpha=q['lora_alpha']).to(device)
            emit('loaded', **model.trainable_parameter_summary())
            inp = prepare_messages(tok,[{'role':'user','content':'Inspect available evidence before proposing a protein edit. If no structure is available, do not invent a residue position.'}])
            target = tok('{"name":"lookup_reaction_evidence","arguments":{},"state_hash":"test"}',return_tensors='pt',add_special_tokens=False)['input_ids']
            model.eval()
            a = model(**inp, target_ids=target)
            b = model(**inp, target_ids=target.flip(-1))
            check('teacher_answer_cannot_change_H', torch.equal(a.hidden,b.hidden))
            check('sft_loss_finite',torch.isfinite(a.language_loss),loss=float(a.language_loss.detach()))
            a.language_loss.backward()
            grad = sum(float(v.grad.detach().float().square().sum()) for n,v in model.named_parameters() if 'lora_' in n and v.grad is not None)**.5
            check('real_Qwen_LoRA_receives_SFT_gradient',grad>0,gradient_norm=grad)
            check('frozen_base_has_no_gradient',all(v.grad is None for n,v in model.named_parameters() if 'lora_' not in n and n!='condition_queries'))
        elif args.component == 'native':
            from agent_dplmbit.generator import load_native_bridge
            from agent_dplmbit.sampling import native_complete, native_feedback_complete, KEEP_CONDITION
            from agent_dplmbit.decisions import DecisionReadout, compile_soft
            model = load_native_bridge(g['model_path'], tokenizer_path=g['tokenizer_path'],
                struct_tokenizer_path=g['struct_tokenizer_path'],condition_dim=27,
                layer_indices=g['layer_indices'],device=device,dtype=torch.bfloat16,raw_condition_dim=4096).eval()
            n=64
            struct,aa=model.encode_state([[-1]*n],[[[-1]*model.bit_dim]*n])
            head=DecisionReadout(4096,model.hidden_size).to(device=device,dtype=torch.bfloat16)
            raw=torch.randn(1,8,4096,device=device,dtype=torch.bfloat16)
            cond=compile_soft(head(raw),n,raw_tokens=raw)
            with torch.no_grad():
                baseline=model(struct,aa,disable_condition=True)
                zero=model(struct,aa,cond.tokens,cond.aa_bias,cond.strength,
                           per_slot_condition=cond.slot_tokens,raw_condition_tokens=raw)
            check('zero_init_AA_matches_native',torch.equal(baseline.aa_logits,zero.aa_logits))
            check('zero_init_bits_match_native',torch.equal(baseline.bit_logits,zero.bit_logits))
            torch.manual_seed(717)
            first=native_complete(model,struct,aa,None,max_iter=16)
            rng1=torch.cuda.get_rng_state().clone()
            hits=[]
            def callback(snapshot):
                hits.append(snapshot.completed_steps)
                # Agent/tool RNG use must not disturb generation RNG.
                torch.randn(7,device=device)
                return KEEP_CONDITION
            torch.manual_seed(717)
            second=native_feedback_complete(model,struct,aa,None,max_iter=16,
                feedback_steps=(4,8,12),feedback_callback=callback)
            rng2=torch.cuda.get_rng_state()
            check('real_native_noop_feedback_token_parity',torch.equal(first.native_tokens,second.native_tokens))
            check('real_native_noop_feedback_rng_parity',torch.equal(rng1,rng2))
            check('three_callbacks_without_schedule_restart',hits==[4,8,12],hits=hits,native_steps=second.native_steps)
        else:
            from agent_dplmbit.train import load_system
            from agent_dplmbit.native_training import sample_native_batch,slice_native_batch,public_from_native_batch,native_training_loss
            from agent_dplmbit.schema import PublicRequest
            from agent_dplmbit.optim import configure_stage,make_optimizer,gradient_norms
            if not args.source:
                raise ValueError('joint probe requires audited natural source')
            rows=[json.loads(x) for x in Path(args.source).read_text().splitlines() if x.strip()]
            row=rows[0]
            model=load_system(config)
            model.eval()
            emit('loaded',component='joint',record=row['parent_id'])
            from agent_dplmbit.decisions import AA_ALPHABET
            aas=torch.tensor([[AA_ALPHABET.index(x) for x in row['sequence']]],device=device)
            bits=torch.tensor([row['bits']],device=device)
            batch=sample_native_batch(model.generator,aas,bits,weighting='constant')
            request=PublicRequest(row['function_text'],len(row['sequence']))
            public=public_from_native_batch(model.generator,batch,0,request)
            losses=[]
            for stage,steps in [('B_bridge',3),('C_joint',3)]:
                groups=configure_stage(model.policy,model.decisions,model.generator,stage)
                model.train()
                opt=make_optimizer(groups,config['learning_rates'])
                for step in range(steps):
                    opt.zero_grad(set_to_none=True)
                    policy,cond,(struct,aa)=model.decisions_for(public)
                    output=model.generator(struct,aa,cond.tokens,cond.aa_bias,cond.strength,
                        per_slot_condition=cond.slot_tokens,raw_condition_tokens=cond.raw_tokens,
                        single_modality=batch.single_modality_index)
                    values=native_training_loss(output,batch,bit_reduction='native_sum')
                    values['loss'].backward()
                    norms=gradient_norms(groups)
                    torch.nn.utils.clip_grad_norm_([v for vs in groups.values() for v in vs],1.,error_if_nonfinite=True)
                    opt.step()
                    loss=float(values['loss'].detach());losses.append(loss)
                    emit('train_probe',stage=stage,step=step+1,loss=loss,gradients=norms)
                    check(f'{stage}_{step}_finite',all(torch.isfinite(x) for x in values.values()))
                check(stage+'_adapter_gradient',norms.get('generator_adapter',0)>0)
                if stage=='C_joint':
                    check('joint_generation_gradient_reaches_Qwen',norms.get('policy',0)>0,gradient_norm=norms.get('policy',0))
                    check('joint_generation_gradient_reaches_DPLM_LoRA',norms.get('generator_lora',0)>0)
            receipt['fixed_batch_losses']=losses
            receipt['scope']='real_weight_fixed_batch_gradient_probe_not_agent_training_or_quality'
        receipt['passed']=True
    except Exception as e:
        receipt['error']=str(e)
        receipt['traceback']=traceback.format_exc()
        traceback.print_exc()
    finally:
        receipt['seconds']=time.time()-started
        if torch.cuda.is_available():
            receipt['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2,allow_nan=False));tmp.replace(path)
        emit('finished',passed=receipt['passed'],output=str(path))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
