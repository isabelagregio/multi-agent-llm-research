#!/usr/bin/env python3
import argparse, csv, json, math, multiprocessing, os, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse as mparse, verify as mverify

ROOT=Path("/lustre06/project/6087921/isa1909")
MODEL=ROOT/"models/Qwen2.5-7B-Instruct"
B_DIR=ROOT/"results/experiment_7_round_scaling_difficulty"
B_RAW=B_DIR/"raw_results.jsonl"
QUESTIONS=B_DIR/"sampled_questions.json"
OUT=ROOT/"results/experiment_G_risk_gated_adaptive_debate"
TEMP=.7; TOP_P=.95; MAX_NEW=512; MAX_LEN=8192; MAX_SEQS=64; GPU_UTIL=.90
JUDGE_SEED=20261023; BOOT=2000; BOOT_SEED=20261024
FINAL_RE=re.compile(r"FINAL:\s*(.+?)(?:\n|$)",re.I)
tok=None; llm=None

ANCHOR_SYSTEM="""You are a risk-aware final adjudicator for a mathematical multi-agent system.
You receive the original problem, two INDEPENDENT anchor solutions produced before communication,
and later revisions produced after interaction.
Independently determine the mathematically correct answer.
Later consensus is NOT evidence by itself. Do not prefer a later answer merely because both agents agree.
A correct independent solution may have been lost through peer influence, while interaction may also
correct earlier errors. Compare the mathematics across the full trajectory, verify key calculations and
logic yourself, preserve a valid earlier argument if later revisions introduce an error, and do not count
repeated versions of the same answer as independent votes.
End with exactly:
FINAL: <answer>""".strip()

METHODS=["debate_r0_current","debate_r1_current","debate_r2_current",
         "adaptive_current","anchor_r1","anchor_r2","rgad"]

def loadj(p):
    with open(p,encoding="utf-8") as f:return json.load(f)
def loadjl(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def savej(p,x):
    t=Path(str(p)+".tmp")
    with open(t,"w",encoding="utf-8") as f:json.dump(x,f,indent=2,ensure_ascii=False)
    os.replace(t,p)
def appendjl(p,rows):
    with open(p,"a",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def writecsv(p,rows):
    if not rows:return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:keys.append(k);seen.add(k)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def final_text(t):
    if not t:return None
    m=FINAL_RE.findall(t);return m[-1].strip() if m else t.strip()
def otext(o):return str(o.get("text") or "") if isinstance(o,dict) else str(o or "")
def oans(o):
    if isinstance(o,dict):
        return str(o["answer"]) if o.get("answer") is not None else final_text(str(o.get("text") or ""))
    return final_text(str(o or ""))
def otok(o):return int(o.get("total_tokens") or 0) if isinstance(o,dict) else 0

def pmath(x):
    if x is None:return []
    v=str(x).strip()
    if not v:return []
    if not ((v.startswith("$") and v.endswith("$")) or "\\boxed" in v):v=f"${v}$"
    return mparse(v,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()])
def match(a,b):
    try:
        pa,pb=pmath(a),pmath(b);return bool(pa and pb and mverify(pb,pa))
    except Exception:return False
def equiv(a,b):
    if a is None or b is None:return False
    try:
        pa,pb=pmath(a),pmath(b)
        if pa and pb:return bool(mverify(pa,pb))
    except Exception:pass
    n=lambda x:re.sub(r"\s+","",str(x).strip().lower())
    return n(a)==n(b)

def init_model():
    global tok,llm
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=True)
    llm=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype="bfloat16",max_model_len=MAX_LEN,
            gpu_memory_utilization=GPU_UTIL,max_num_seqs=MAX_SEQS,
            enable_prefix_caching=True,trust_remote_code=True)
def render(msg):
    p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    n=len(tok(p,add_special_tokens=False).input_ids)
    if n+MAX_NEW>MAX_LEN:raise ValueError(f"Prompt too long: {n}+{MAX_NEW}>{MAX_LEN}")
    return p
def generate(msgs,seed):
    ps=[render(x) for x in msgs]
    sp=SamplingParams(n=1,temperature=TEMP,top_p=TOP_P,max_tokens=MAX_NEW,seed=seed)
    st=time.perf_counter(); outs=llm.generate(ps,sp,use_tqdm=True); wall=time.perf_counter()-st
    ret=[]
    for ro in outs:
        pt=len(getattr(ro,"prompt_token_ids",[]) or []); c=ro.outputs[0]; ct=len(c.token_ids)
        ret.append({"text":c.text,"answer":final_text(c.text),"prompt_tokens":pt,
                    "completion_tokens":ct,"total_tokens":pt+ct})
    return ret,wall

def st(row,r):return row["debate"][str(r)]
def agents(row,r):return oans(st(row,r)["agent_a"]),oans(st(row,r)["agent_b"])
def texts(row,r):return otext(st(row,r)["agent_a"]),otext(st(row,r)["agent_b"])
def path_tokens(row,r):
    return sum(otok(st(row,k)["agent_a"])+otok(st(row,k)["agent_b"]) for k in range(r+1))
def current(row,r):
    return {"answer":oans(st(row,r)["judge"]),"correct":bool(st(row,r)["correct"]),"rounds":r,
            "tokens":path_tokens(row,r)+otok(st(row,r)["judge"])}

def policy(row):
    a0,b0=agents(row,0)
    if equiv(a0,b0):return {"stop_round":0,"stop_reason":"initial_consensus"}
    a1,b1=agents(row,1)
    if equiv(a1,b1):return {"stop_round":1,"stop_reason":"new_consensus_after_disagreement"}
    if equiv(a0,a1) and equiv(b0,b1):return {"stop_round":1,"stop_reason":"stagnant_disagreement"}
    return {"stop_round":2,"stop_reason":"active_unresolved_disagreement"}

def judge_msg(problem,row,r):
    a0,b0=texts(row,0)
    parts=[f"ORIGINAL PROBLEM\n{problem}",f"INDEPENDENT ANCHOR A0\n{a0}",f"INDEPENDENT ANCHOR B0\n{b0}"]
    for k in range(1,r+1):
        a,b=texts(row,k);parts += [f"POST-INTERACTION A{k}\n{a}",f"POST-INTERACTION B{k}\n{b}"]
    return [{"role":"system","content":ANCHOR_SYSTEM},{"role":"user","content":"\n\n".join(parts)}]

def anchor_result(row,r,o):
    return {"answer":o["answer"],"correct":match(o["answer"],row["gold"]),"rounds":r,
            "tokens":path_tokens(row,r)+o["total_tokens"]}
def adaptive_current(row,d):
    r=d["stop_round"]
    if r==0:
        a,_=agents(row,0);return {"answer":a,"correct":match(a,row["gold"]),"rounds":0,"tokens":path_tokens(row,0)}
    return current(row,r)
def rgad(row,d,a1,a2):
    r=d["stop_round"]
    if r==0:
        a,_=agents(row,0);return {"answer":a,"correct":match(a,row["gold"]),"rounds":0,"tokens":path_tokens(row,0)}
    return anchor_result(row,r,a1 if r==1 else a2)

def risk_features(row):
    def ac(r):
        a,b=agents(row,r);return match(a,row["gold"]),match(b,row["gold"])
    a0,b0=ac(0);a1,b1=ac(1);a2,b2=ac(2)
    x0,y0=agents(row,0);x1,y1=agents(row,1);x2,y2=agents(row,2)
    d0=not equiv(x0,y0);d1=not equiv(x1,y1)
    return {
      "contamination_r1":(a0!=b0 and not a1 and not b1),
      "contamination_r2":(a1!=b1 and not a2 and not b2),
      "harmful_consensus_r1":(d0 and equiv(x1,y1) and not a1 and not b1),
      "harmful_consensus_r2":(d1 and equiv(x2,y2) and not a2 and not b2),
      "any_correct_through_r1":any([a0,b0,a1,b1]),
      "any_correct_through_r2":any([a0,b0,a1,b1,a2,b2]),
      "current_judge_r1_correct":bool(st(row,1)["correct"]),
      "current_judge_r2_correct":bool(st(row,2)["correct"])
    }

def summarize(rows,rep,wall):
    m={}
    for name in METHODS:
        m[name]={"accuracy":mean(1 if x["methods"][name]["correct"] else 0 for x in rows),
                 "mean_tokens_per_question":mean(x["methods"][name]["tokens"] for x in rows),
                 "mean_rounds":mean(x["methods"][name]["rounds"] for x in rows)}
    c=Counter(x["policy"]["stop_round"] for x in rows); rs=Counter(x["policy"]["stop_reason"] for x in rows)
    return {"repetition":rep,"questions":len(rows),"methods":m,
            "stop_distribution":{str(r):{"count":c[r],"fraction":c[r]/len(rows)} for r in [0,1,2]},
            "stop_reasons":dict(rs),"anchor_judge_generation_wall_seconds":wall}

def run_rep(allrows,problems,rep):
    rows=sorted([x for x in allrows if int(x["repetition"])==rep],key=lambda x:int(x["question_id"]))
    dec=[policy(x) for x in rows]
    print("stop reasons:",dict(Counter(x["stop_reason"] for x in dec)),flush=True)
    a1,w1=generate([judge_msg(problems[int(x["question_id"])],x,1) for x in rows],JUDGE_SEED+rep*10000+1)
    a2,w2=generate([judge_msg(problems[int(x["question_id"])],x,2) for x in rows],JUDGE_SEED+rep*10000+2)
    out=[]
    for i,row in enumerate(rows):
        d=dec[i]; risk=risk_features(row)
        risk["anchor_judge_r1_correct"]=match(a1[i]["answer"],row["gold"])
        risk["anchor_judge_r2_correct"]=match(a2[i]["answer"],row["gold"])
        methods={"debate_r0_current":current(row,0),"debate_r1_current":current(row,1),
                 "debate_r2_current":current(row,2),"adaptive_current":adaptive_current(row,d),
                 "anchor_r1":anchor_result(row,1,a1[i]),"anchor_r2":anchor_result(row,2,a2[i]),
                 "rgad":rgad(row,d,a1[i],a2[i])}
        out.append({"question_id":int(row["question_id"]),"repetition":rep,"level":int(row["level"]),
                    "type":row.get("type"),"policy":d,"methods":methods,"risk_cases":risk,
                    "anchor_judge_r1":a1[i],"anchor_judge_r2":a2[i]})
    return out,summarize(out,rep,w1+w2)

def percentile(v,q):
    v=sorted(v);p=(len(v)-1)*q;l=math.floor(p);h=math.ceil(p)
    return v[l] if l==h else v[l]*(h-p)+v[h]*(p-l)
def bootstrap(rows):
    by=defaultdict(list)
    for x in rows:by[x["question_id"]].append(x)
    ids=list(by);rng=random.Random(BOOT_SEED); baselines=[x for x in METHODS if x!="rgad"]
    vals={b:{"acc":[],"save":[]} for b in baselines}
    for _ in range(BOOT):
        samp=[]
        for q in [rng.choice(ids) for _ in ids]:samp.extend(by[q])
        ra=mean(1 if x["methods"]["rgad"]["correct"] else 0 for x in samp)
        rt=mean(x["methods"]["rgad"]["tokens"] for x in samp)
        for b in baselines:
            ba=mean(1 if x["methods"][b]["correct"] else 0 for x in samp)
            bt=mean(x["methods"][b]["tokens"] for x in samp)
            vals[b]["acc"].append(ra-ba);vals[b]["save"].append((bt-rt)/bt)
    return vals

def aggregate(rows,reps):
    ms=[]
    for name in METHODS:
        ms.append({"method":name,"accuracy":mean(1 if x["methods"][name]["correct"] else 0 for x in rows),
                   "mean_tokens_per_question":mean(x["methods"][name]["tokens"] for x in rows),
                   "mean_rounds":mean(x["methods"][name]["rounds"] for x in rows)})
    md={x["method"]:x for x in ms};boot=bootstrap(rows);pairs=[]
    for b,v in boot.items():
        alo,ahi=percentile(v["acc"],.025),percentile(v["acc"],.975)
        slo,shi=percentile(v["save"],.025),percentile(v["save"],.975)
        pairs.append({"comparison":f"rgad - {b}","accuracy_difference":md["rgad"]["accuracy"]-md[b]["accuracy"],
                      "accuracy_diff_ci95_low":alo,"accuracy_diff_ci95_high":ahi,
                      "token_savings_fraction":(md[b]["mean_tokens_per_question"]-md["rgad"]["mean_tokens_per_question"])/md[b]["mean_tokens_per_question"],
                      "token_savings_ci95_low":slo,"token_savings_ci95_high":shi})
    risk=[]
    for r in [1,2]:
        for key in [f"contamination_r{r}",f"harmful_consensus_r{r}"]:
            sub=[x for x in rows if x["risk_cases"][key]]
            if sub:
                ca=mean(1 if x["risk_cases"][f"current_judge_r{r}_correct"] else 0 for x in sub)
                aa=mean(1 if x["risk_cases"][f"anchor_judge_r{r}_correct"] else 0 for x in sub)
                risk.append({"risk_case":key,"round":r,"n":len(sub),"current_judge_accuracy":ca,
                             "anchor_judge_accuracy":aa,"anchor_minus_current":aa-ca})
        sub=[x for x in rows if x["risk_cases"][f"any_correct_through_r{r}"]]
        ca=mean(1 if x["risk_cases"][f"current_judge_r{r}_correct"] else 0 for x in sub)
        aa=mean(1 if x["risk_cases"][f"anchor_judge_r{r}_correct"] else 0 for x in sub)
        risk.append({"risk_case":f"correct_candidate_available_r{r}","round":r,"n":len(sub),
                     "current_judge_accuracy":ca,"anchor_judge_accuracy":aa,"anchor_minus_current":aa-ca})
    sc=Counter(x["policy"]["stop_round"] for x in rows);sr=Counter(x["policy"]["stop_reason"] for x in rows)
    stops=[]
    for reason,n in sr.items():
        sub=[x for x in rows if x["policy"]["stop_reason"]==reason]
        stops.append({"stop_reason":reason,"count":n,"fraction":n/len(rows),
                      "rgad_accuracy":mean(1 if x["methods"]["rgad"]["correct"] else 0 for x in sub),
                      "mean_rounds":mean(x["methods"]["rgad"]["rounds"] for x in sub)})
    return {"experiment":"G - Risk-Gated Adaptive Debate","method_name":"RGAD",
            "research_question":"Can adaptive interaction use observable utility/risk signals while preserving independent anchors to reduce cost and error propagation?",
            "design":{"trajectory_source":"Experiment B Debate traces","questions":len(set(x["question_id"] for x in rows)),
                      "repetitions":len(set(x["repetition"] for x in rows)),"max_rounds":2,"policy_uses_gold":False,
                      "new_inference":"Anchor-aware R1/R2 judges only","bootstrap":"clustered by question_id"},
            "policy":{"R0":"stop on initial consensus",
                      "R1":"stop on newly formed consensus or stagnant disagreement; otherwise continue",
                      "R2":"mandatory stop","selection":"anchor-aware judge sees independent anchors and revisions"},
            "method_summary":ms,"paired_rgad_comparisons":pairs,
            "stop_distribution":{str(r):{"count":sc[r],"fraction":sc[r]/len(rows)} for r in [0,1,2]},
            "stop_reason_summary":stops,"risk_case_analysis":risk,"repetition_summaries":reps}

def plots(final):
    md={x["method"]:x for x in final["method_summary"]}
    show=["debate_r0_current","debate_r1_current","debate_r2_current","adaptive_current","anchor_r1","rgad"]
    plt.figure(figsize=(8,5))
    for name in show:
        x=md[name]["mean_tokens_per_question"];y=100*md[name]["accuracy"]
        plt.scatter(x,y,s=70);plt.annotate(name,(x,y),xytext=(5,5),textcoords="offset points")
    plt.xlabel("Mean tokens per question");plt.ylabel("Accuracy (%)");plt.title("Experiment G — Accuracy vs. Token Cost")
    plt.grid(alpha=.2);plt.tight_layout();plt.savefig(OUT/"accuracy_vs_tokens.png",dpi=300);plt.close()
    rr=[x for x in final["risk_case_analysis"] if x["risk_case"].startswith(("contamination","harmful_consensus"))]
    pos=list(range(len(rr)));w=.38
    plt.figure(figsize=(9,5));plt.bar([i-w/2 for i in pos],[100*x["current_judge_accuracy"] for x in rr],w,label="Current-state judge")
    plt.bar([i+w/2 for i in pos],[100*x["anchor_judge_accuracy"] for x in rr],w,label="Anchor-aware judge")
    plt.xticks(pos,[x["risk_case"] for x in rr],rotation=25,ha="right");plt.ylabel("Recovery accuracy (%)")
    plt.title("Experiment G — Recovery on Error-Propagation Cases");plt.legend();plt.tight_layout()
    plt.savefig(OUT/"risk_recovery.png",dpi=300);plt.close()
    vals=[100*final["stop_distribution"][str(r)]["fraction"] for r in [0,1,2]]
    plt.figure(figsize=(6,4));bars=plt.bar(["R0","R1","R2"],vals);plt.ylabel("Questions stopped (%)")
    plt.title("Experiment G — RGAD Stop Distribution")
    for b,v in zip(bars,vals):plt.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}%",ha="center",va="bottom")
    plt.tight_layout();plt.savefig(OUT/"stop_distribution.png",dpi=300);plt.close()

def quick(final):
    lines=["EXPERIMENT G — RISK-GATED ADAPTIVE DEBATE (RGAD)",""]
    for x in final["method_summary"]:
        lines.append(f"{x['method']:<22} acc={100*x['accuracy']:.2f}% tokens/q={x['mean_tokens_per_question']:.1f} rounds={x['mean_rounds']:.3f}")
    lines+=["","RGAD PAIRED COMPARISONS"]
    for x in final["paired_rgad_comparisons"]:
        lines.append(f"{x['comparison']}: acc={100*x['accuracy_difference']:+.2f}pp [{100*x['accuracy_diff_ci95_low']:+.2f},{100*x['accuracy_diff_ci95_high']:+.2f}] token_savings={100*x['token_savings_fraction']:+.2f}%")
    lines+=["","RISK CASES"]
    for x in final["risk_case_analysis"]:
        lines.append(f"{x['risk_case']}: n={x['n']} current={100*x['current_judge_accuracy']:.2f}% anchor={100*x['anchor_judge_accuracy']:.2f}% delta={100*x['anchor_minus_current']:+.2f}pp")
    (OUT/"quick_read.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    global B_RAW,QUESTIONS,OUT
    ap=argparse.ArgumentParser();ap.add_argument("--b-raw",type=Path,default=B_RAW)
    ap.add_argument("--questions",type=Path,default=QUESTIONS);ap.add_argument("--output-dir",type=Path,default=OUT)
    a=ap.parse_args();B_RAW=a.b_raw;QUESTIONS=a.questions;OUT=a.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    if not B_RAW.exists():raise FileNotFoundError(B_RAW)
    if not QUESTIONS.exists():raise FileNotFoundError(QUESTIONS)
    rows=loadjl(B_RAW);qs=loadj(QUESTIONS);problems={int(q["question_id"]):q["problem"] for q in qs}
    reps=sorted(set(int(x["repetition"]) for x in rows))
    print("Experiment G — RGAD",len(rows),"rows",reps,flush=True);init_model()
    raw=OUT/"raw_results.jsonl"
    if raw.exists():raw.unlink()
    allout=[];summ=[]
    for rep in reps:
        rr,ss=run_rep(rows,problems,rep);appendjl(raw,rr);savej(OUT/f"repetition_{rep}_summary.json",ss)
        allout+=rr;summ.append(ss)
    final=aggregate(allout,summ);savej(OUT/"final_summary.json",final)
    writecsv(OUT/"method_summary.csv",final["method_summary"]);writecsv(OUT/"paired_comparisons.csv",final["paired_rgad_comparisons"])
    writecsv(OUT/"risk_case_analysis.csv",final["risk_case_analysis"]);writecsv(OUT/"stop_reason_summary.csv",final["stop_reason_summary"])
    plots(final);quick(final);print((OUT/"quick_read.txt").read_text(),flush=True)

if __name__=="__main__":
    multiprocessing.freeze_support();main()
