#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",required=True)
    ap.add_argument("--out",default="")
    args=ap.parse_args()
    s=json.loads(Path(args.summary).read_text(encoding="utf-8"))
    pre=s.get("pre_audit",{}); post=s.get("post_audit",{})
    keys=["foot_skate_mean_mpf","foot_penetration_min_m","mean_joint_jerk_p95","mean_joint_jerk_max","collision_risk","root_y_acc_p95"]
    out={"version":"v38_contact_jitter_delta","summary":args.summary,"planner_feedback":s.get("planner_feedback",{})}
    for k in keys:
        out[k]={"pre":float(pre.get(k,0.0)),"post":float(post.get(k,0.0)),"delta":float(pre.get(k,0.0)-post.get(k,0.0))}
    txt=json.dumps(out,ensure_ascii=False,indent=2)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(txt,encoding="utf-8")
if __name__=="__main__": main()
