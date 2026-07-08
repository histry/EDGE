#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that V46.38 report actually used final MSSD and MSSD-AESD routing."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--mssd', default='')
    a=ap.parse_args()
    r=json.load(open(a.report, encoding='utf-8'))
    retrieval=r.get('stage_reports',{}).get('retrieval',[])
    if not retrieval:
        raise RuntimeError('no retrieval report in '+a.report)
    policies=[str(x.get('routing_policy','')) for x in retrieval]
    if not any('V46.38 MSSD-AESD' in p for p in policies):
        raise RuntimeError('routing_policy does not show V46.38 MSSD-AESD')
    previews=[x.get('candidate_preview',[]) for x in retrieval]
    flat=[y for p in previews for y in p]
    required=['mssd_aesd_semantic_score','contrastive_similarity','natural_duration_score','aesd_boundary_risk']
    miss=[k for k in required if not any(k in y for y in flat)]
    if miss:
        raise RuntimeError('candidate_preview missing: '+str(miss))
    mssd_ok=0
    for x in retrieval:
        aud=x.get('mssd_audit',{})
        if aud.get('mssd_is_final_schedule') is True or aud.get('is_final_schedule') is True:
            mssd_ok+=1
    if mssd_ok == 0:
        raise RuntimeError('no slot records carry final MSSD audit fields')
    out={'report':a.report,'slots':len(retrieval),'v46_38_policy':True,'final_mssd_slots_with_audit':mssd_ok,'sample_policy':policies[0]}
    if a.mssd:
        d=json.load(open(a.mssd, encoding='utf-8'))
        out['mssd']={k:d.get(k) for k in ['usage','is_final_schedule','slot_source','num_slots','total_target_frames','router_ckpt','planner_ckpt','v23_ckpt','raw_schedule_json']}
        if d.get('usage')!='generate_schedule' or d.get('is_final_schedule') is not True:
            raise RuntimeError('MSSD is not strict generate_schedule')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
