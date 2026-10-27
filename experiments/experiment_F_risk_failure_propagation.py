#!/usr/bin/env python3
import argparse, csv, json, math, random
from collections import defaultdict
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse as math_parse, verify as math_verify
except ImportError as exc:
    raise SystemExit("Install math-verify with: pip install 'math-verify[antlr4_13_2]'") from exc

ROOT = Path("/lustre06/project/6087921/isa1909")
OUT = ROOT / "results" / "experiment_F_risk_failure_propagation"
D_RAW = ROOT / "results" / "experiment_D_adaptive_stopping" / "raw_results.jsonl"
B_CANDIDATES = [
    ROOT / "results" / "experiment_7_round_scaling_difficulty" / "raw_results.jsonl",
    ROOT / "results" / "experiment_B_round_scaling_difficulty" / "raw_results.jsonl",
    ROOT / "results" / "experiment_b_round_scaling_difficulty" / "raw_results.jsonl",
]
ROUNDS = [0,1,2,3]
BOOT = 2000
SEED = 20261019

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_json(p, x):
    with open(p,"w",encoding="utf-8") as f: json.dump(x,f,indent=2,ensure_ascii=False)

def write_csv(p, rows):
    if not rows: return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def rate(a,b): return a/b if b else None
def trans(a,b):
    return "C->C" if a and b else "C->W" if a else "W->C" if b else "W->W"

def parse_math(x):
    if x is None: return []
    s=str(x).strip()
    if not s: return []
    if not ((s.startswith("$") and s.endswith("$")) or "\\boxed" in s): s=f"${s}$"
    return math_parse(s, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])

def match(pred,gold):
    try:
        p,g=parse_math(pred),parse_math(gold)
        return bool(p and g and math_verify(g,p))
    except Exception:
        return False

def answer(o):
    if isinstance(o,dict): return o.get("answer",o.get("text"))
    return o

def percentile(xs,q):
    xs=sorted(xs); pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi:return xs[lo]
    return xs[lo]*(hi-pos)+xs[hi]*(pos-lo)

def bootstrap(rows, metric, seed):
    by=defaultdict(list)
    for r in rows: by[r["question_id"]].append(r)
    qids=list(by); rng=random.Random(seed); vals=[]
    for _ in range(BOOT):
        sample=[]
        for q in [rng.choice(qids) for _ in qids]: sample.extend(by[q])
        v=metric(sample)
        if v is not None: vals.append(v)
    return (percentile(vals,.025),percentile(vals,.975)) if vals else (None,None)

def sequential(rows,method):
    rec=[]
    for row in rows:
        s=row[method]
        for r in range(1,4):
            a=bool(s[str(r-1)]["correct"]); b=bool(s[str(r)]["correct"])
            rec.append(dict(question_id=row["question_id"],repetition=row["repetition"],level=row["level"],
                            method=method,round=r,before=a,after=b,transition=trans(a,b)))
    out=[]
    for r in range(1,4):
        x=[z for z in rec if z["round"]==r]
        cb=sum(z["before"] for z in x); wb=len(x)-cb
        harm=sum(z["transition"]=="C->W" for z in x); help_=sum(z["transition"]=="W->C" for z in x)
        hlo,hhi=bootstrap(x,lambda y:rate(sum(z["transition"]=="C->W" for z in y),sum(z["before"] for z in y)),SEED+r)
        blo,bhi=bootstrap(x,lambda y:rate(sum(z["transition"]=="W->C" for z in y),sum(not z["before"] for z in y)),SEED+100+r)
        out.append(dict(method=method,round=r,n=len(x),correct_before=cb,wrong_before=wb,
                        harmful_count=harm,harmful_rate_given_correct=rate(harm,cb),
                        harmful_ci95_low=hlo,harmful_ci95_high=hhi,
                        beneficial_count=help_,beneficial_rate_given_wrong=rate(help_,wb),
                        beneficial_ci95_low=blo,beneficial_ci95_high=bhi,
                        net_corrections=help_-harm))
    return rec,out

def student_critic(rows):
    rec=[]
    for row in rows:
        s=row["student_critic"]
        for r in range(1,4):
            a=bool(s[str(r-1)]["correct"]); b=bool(s[str(r)]["correct"])
            v=(s[str(r)].get("verdict") or "UNKNOWN").upper()
            rec.append(dict(question_id=row["question_id"],repetition=row["repetition"],level=row["level"],round=r,
                before=a,after=b,verdict=v,transition=trans(a,b),
                false_reject=(v=="REJECT" and a),
                harmful_reject=(v=="REJECT" and a and not b),
                false_accept=(v=="ACCEPT" and not a),
                useful_reject=(v=="REJECT" and not a and b)))
    out=[]
    for r in range(1,4):
        x=[z for z in rec if z["round"]==r]
        cb=sum(z["before"] for z in x); wb=len(x)-cb; rejects=sum(z["verdict"]=="REJECT" for z in x)
        fr=sum(z["false_reject"] for z in x); hr=sum(z["harmful_reject"] for z in x)
        fa=sum(z["false_accept"] for z in x); ur=sum(z["useful_reject"] for z in x)
        lo,hi=bootstrap(x,lambda y:rate(sum(z["false_reject"] for z in y),sum(z["before"] for z in y)),SEED+200+r)
        out.append(dict(round=r,n=len(x),reject_count=rejects,false_reject_count=fr,
                        false_reject_rate_given_correct=rate(fr,cb),false_reject_ci95_low=lo,false_reject_ci95_high=hi,
                        harmful_reject_count=hr,harmful_given_false_reject_rate=rate(hr,fr),
                        false_accept_count=fa,false_accept_rate_given_wrong=rate(fa,wb),
                        useful_reject_correction_count=ur,useful_reject_precision=rate(ur,rejects)))
    return rec,out

def debate(rows):
    rec=[]
    for row in rows:
        corr={}
        for r in ROUNDS:
            s=row["debate"][str(r)]
            corr[r]=(match(answer(s["agent_a"]),row["gold"]),match(answer(s["agent_b"]),row["gold"]),bool(s["correct"]))
            a,b,j=corr[r]
            rec.append(dict(question_id=row["question_id"],repetition=row["repetition"],level=row["level"],
                            kind="judge",round=r,one_correct=(a!=b),both_correct=(a and b),both_wrong=(not a and not b),
                            judge_correct=j,judge_error_one=(a!=b and not j),
                            judge_error_both_correct=(a and b and not j),
                            judge_rescue_both_wrong=(not a and not b and j)))
        for r in range(1,4):
            pa,pb,_=corr[r-1]; ca,cb,_=corr[r]
            prev_dis=bool(row["debate"][str(r-1)]["disagreement"]); cur_dis=bool(row["debate"][str(r)]["disagreement"])
            one=(pa!=pb); bothw=(not ca and not cb); bothc=(ca and cb); new_cons=prev_dis and not cur_dis
            rec.append(dict(question_id=row["question_id"],repetition=row["repetition"],level=row["level"],
                            kind="interaction",round=r,one_correct_before=one,new_consensus=new_cons,
                            contamination=(one and bothw),corrective_convergence=(one and bothc),
                            harmful_consensus=(new_cons and bothw),beneficial_consensus=(new_cons and bothc)))
    out=[]
    for r in range(1,4):
        x=[z for z in rec if z["kind"]=="interaction" and z["round"]==r]
        one=sum(z["one_correct_before"] for z in x); nc=sum(z["new_consensus"] for z in x)
        cont=sum(z["contamination"] for z in x); conv=sum(z["corrective_convergence"] for z in x)
        hc=sum(z["harmful_consensus"] for z in x); bc=sum(z["beneficial_consensus"] for z in x)
        lo,hi=bootstrap(x,lambda y:rate(sum(z["contamination"] for z in y),sum(z["one_correct_before"] for z in y)),SEED+400+r)
        out.append(dict(analysis="interaction",round=r,n=len(x),one_correct_before_count=one,
                        correct_agent_contamination_count=cont,correct_agent_contamination_rate=rate(cont,one),
                        contamination_ci95_low=lo,contamination_ci95_high=hi,
                        corrective_convergence_count=conv,corrective_convergence_rate=rate(conv,one),
                        new_consensus_count=nc,harmful_consensus_count=hc,harmful_consensus_rate=rate(hc,nc),
                        beneficial_consensus_count=bc))
    for r in ROUNDS:
        x=[z for z in rec if z["kind"]=="judge" and z["round"]==r]
        one=sum(z["one_correct"] for z in x); bc=sum(z["both_correct"] for z in x); bw=sum(z["both_wrong"] for z in x)
        out.append(dict(analysis="judge",round=r,n=len(x),one_correct_count=one,
                        judge_error_one_correct_count=sum(z["judge_error_one"] for z in x),
                        judge_error_one_correct_rate=rate(sum(z["judge_error_one"] for z in x),one),
                        both_correct_count=bc,judge_error_both_correct_count=sum(z["judge_error_both_correct"] for z in x),
                        judge_error_both_correct_rate=rate(sum(z["judge_error_both_correct"] for z in x),bc),
                        both_wrong_count=bw,judge_rescue_both_wrong_count=sum(z["judge_rescue_both_wrong"] for z in x),
                        judge_rescue_both_wrong_rate=rate(sum(z["judge_rescue_both_wrong"] for z in x),bw)))
    return rec,out

def adaptive(rows):
    out=[]
    for level in ["all",1,2,3,4,5]:
        x=rows if level=="all" else [r for r in rows if r["level"]==level]
        if not x: continue
        prem=sum(bool(r["adaptive"]["premature_stop"]) for r in x)
        avoid=sum(bool(r["adaptive"]["avoided_later_degradation"]) for r in x)
        early=sum(int(r["adaptive"]["stop_round"])<3 for r in x)
        out.append(dict(level=level,n=len(x),stopped_before_r3_rate=rate(early,len(x)),
                        premature_stop_count=prem,premature_stop_rate=rate(prem,len(x)),
                        avoided_later_degradation_count=avoid,avoided_later_degradation_rate=rate(avoid,len(x)),
                        risk_benefit_balance_count=avoid-prem))
    return out

def by_difficulty(self_rec,critic_rec,deb_rec):
    out=[]
    for l in range(1,6):
        for name,recs in [("self_refine",self_rec),("student_critic",critic_rec)]:
            x=[z for z in recs if z["level"]==l]
            cb=sum(z["before"] for z in x); wb=len(x)-cb
            h=sum(z["transition"]=="C->W" for z in x); b=sum(z["transition"]=="W->C" for z in x)
            out.append(dict(level=l,method=name,harmful_rate_given_correct=rate(h,cb),
                            beneficial_rate_given_wrong=rate(b,wb),n=len(x)))
        x=[z for z in deb_rec if z["kind"]=="interaction" and z["level"]==l]
        one=sum(z["one_correct_before"] for z in x); nc=sum(z["new_consensus"] for z in x)
        out.append(dict(level=l,method="debate",correct_agent_contamination_rate=rate(sum(z["contamination"] for z in x),one),
                        harmful_consensus_rate=rate(sum(z["harmful_consensus"] for z in x),nc),n=len(x)))
    return out

def fmt(v): return "NA" if v is None else f"{100*v:.2f}%"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--b-raw",type=Path,default=None)
    ap.add_argument("--d-raw",type=Path,default=D_RAW)
    ap.add_argument("--output-dir",type=Path,default=OUT)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    braw=a.b_raw or next((p for p in B_CANDIDATES if p.exists()),None)
    if braw is None: raise FileNotFoundError("Could not find Experiment B raw_results.jsonl; pass --b-raw PATH")
    B=load_jsonl(braw); D=load_jsonl(a.d_raw) if a.d_raw.exists() else []
    srec,ssum=sequential(B,"self_refine")
    crec,ctsum=sequential(B,"student_critic")
    crrec,crsum=student_critic(B)
    drec,dsum=debate(B)
    asum=adaptive(D) if D else []
    dif=by_difficulty(srec,crrec,drec)
    write_csv(a.output_dir/"transition_risk.csv",ssum+ctsum)
    write_csv(a.output_dir/"student_critic_risk.csv",crsum)
    write_csv(a.output_dir/"debate_risk.csv",dsum)
    write_csv(a.output_dir/"adaptive_risk.csv",asum)
    write_csv(a.output_dir/"risk_by_difficulty.csv",dif)

    labels=[f"{x['method']} R{x['round']}" for x in ssum+ctsum]
    harm=[100*(x["harmful_rate_given_correct"] or 0) for x in ssum+ctsum]
    ben=[100*(x["beneficial_rate_given_wrong"] or 0) for x in ssum+ctsum]
    X=list(range(len(labels))); w=.38
    plt.figure(figsize=(11,5)); plt.bar([i-w/2 for i in X],harm,w,label="Correct -> wrong")
    plt.bar([i+w/2 for i in X],ben,w,label="Wrong -> correct"); plt.xticks(X,labels,rotation=35,ha="right")
    plt.ylabel("Conditional transition rate (%)"); plt.title("Experiment F — Harmful vs. Beneficial Transitions")
    plt.legend(); plt.tight_layout(); plt.savefig(a.output_dir/"harmful_vs_beneficial_transitions.png",dpi=300); plt.close()

    ir=[x for x in dsum if x["analysis"]=="interaction"]; rr=[x["round"] for x in ir]
    plt.figure(figsize=(7,4.5)); plt.plot(rr,[100*(x["correct_agent_contamination_rate"] or 0) for x in ir],marker="o",label="Correct-agent contamination")
    plt.plot(rr,[100*(x["corrective_convergence_rate"] or 0) for x in ir],marker="o",label="Corrective convergence")
    plt.xticks(rr); plt.xlabel("Interaction round"); plt.ylabel("Rate (%)"); plt.title("Experiment F — Debate Error Propagation")
    plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(a.output_dir/"debate_contagion_by_round.png",dpi=300); plt.close()

    rr=[x["round"] for x in crsum]
    plt.figure(figsize=(7,4.5)); plt.plot(rr,[100*(x["false_reject_rate_given_correct"] or 0) for x in crsum],marker="o",label="False reject | correct")
    plt.plot(rr,[100*(x["harmful_given_false_reject_rate"] or 0) for x in crsum],marker="o",label="Harm | false reject")
    plt.xticks(rr); plt.xlabel("Critic round"); plt.ylabel("Rate (%)"); plt.title("Experiment F — Student-Critic Risk")
    plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(a.output_dir/"student_critic_false_rejects.png",dpi=300); plt.close()

    final={"experiment":"F - Risk and Failure Propagation Analysis",
           "research_question":"When does LLM interaction correct individual errors, and when does it propagate or amplify them into collective failures?",
           "source_B_raw":str(braw),"source_D_raw":str(a.d_raw) if D else None,
           "B_rows":len(B),"D_rows":len(D),"unique_questions_B":len(set(x["question_id"] for x in B)),
           "repetitions_B":len(set(x["repetition"] for x in B)),
           "self_refine":ssum,"student_critic_transitions":ctsum,"student_critic_critic_risk":crsum,
           "debate":dsum,"adaptive":asum,"risk_by_difficulty":dif,
           "definitions":{"correct_agent_contamination":"Exactly one debate agent is correct before interaction and both are wrong after.",
                          "harmful_consensus":"Agents disagree before interaction and agree on a wrong answer after.",
                          "false_reject":"Critic says REJECT although the Student was already correct.",
                          "premature_stop":"Adaptive stopping ends wrong even though a later fixed checkpoint becomes correct."}}
    write_json(a.output_dir/"final_summary.json",final)

    lines=["EXPERIMENT F — RISK AND FAILURE PROPAGATION",""]
    lines+=["SELF-REFINE"]+[f"R{x['round']}: harm={fmt(x['harmful_rate_given_correct'])} help={fmt(x['beneficial_rate_given_wrong'])} net={x['net_corrections']:+d}" for x in ssum]
    lines+=["","STUDENT-CRITIC"]+[f"R{x['round']}: false-reject={fmt(x['false_reject_rate_given_correct'])} harm|false-reject={fmt(x['harmful_given_false_reject_rate'])}" for x in crsum]
    lines+=["","DEBATE"]+[f"R{x['round']}: contamination={fmt(x['correct_agent_contamination_rate'])} corrective-convergence={fmt(x['corrective_convergence_rate'])} harmful-consensus={fmt(x['harmful_consensus_rate'])}" for x in ir]
    if asum:
        z=next(x for x in asum if x["level"]=="all")
        lines+=["","ADAPTIVE",f"premature-stop={fmt(z['premature_stop_rate'])} avoided-degradation={fmt(z['avoided_later_degradation_rate'])} balance={z['risk_benefit_balance_count']:+d}"]
    (a.output_dir/"quick_read.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print((a.output_dir/"quick_read.txt").read_text())

if __name__=="__main__":
    main()
