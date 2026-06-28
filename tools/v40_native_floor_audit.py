#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V40 native floor pathology audit for Event-RAG libraries.

Reads a V21/V34 JSON index and computes per-event foot-floor pathology from the
stored motion snippets.  It writes an augmented JSON with fields:
  native_min_foot_y, native_floor_y, native_floor_penetration_m,
  native_floor_penalty, native_floor_ok
and optionally filters out severe native penetrators.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],dtype=np.int64)
OFFSETS=np.array([[0,0,0],[-.10,-.10,0],[.10,-.10,0],[0,.13,0],[0,-.42,0],[0,-.42,0],[0,.14,0],[0,-.40,0],[0,-.40,0],[0,.14,0],[0,-.08,.12],[0,-.08,.12],[0,.14,0],[-.10,.08,0],[.10,.08,0],[0,.16,0],[-.18,0,0],[.18,0,0],[-.28,0,0],[.28,0,0],[-.25,0,0],[.25,0,0],[-.08,0,0],[.08,0,0]],dtype=np.float32)
FOOT_JOINTS=np.array([7,8,10,11],dtype=np.int64)

def _rot6d_to_matrix_np(x):
    a1=x[...,0:3]; a2=x[...,3:6]
    b1=a1/np.maximum(np.linalg.norm(a1,axis=-1,keepdims=True),1e-8)
    proj=np.sum(b1*a2,axis=-1,keepdims=True)*b1
    b2=a2-proj; b2=b2/np.maximum(np.linalg.norm(b2,axis=-1,keepdims=True),1e-8)
    b3=np.cross(b1,b2)
    return np.stack([b1,b2,b3],axis=-1).astype(np.float32)

def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def _items(obj):
    if isinstance(obj, list): return obj
    for key in ('items','events','index','data'):
        if isinstance(obj, dict) and isinstance(obj.get(key), list): return obj[key]
    raise ValueError('Cannot locate item list in JSON')

def _find_motion_path(item: Dict[str,Any], roots: Sequence[Path]) -> Optional[Path]:
    keys=('motion_path','npy_path','path','file','motion_file','event_path','source_path','clip_path')
    vals=[]
    for k in keys:
        v=item.get(k)
        if isinstance(v,str) and v:
            vals.append(v)
    # common id fields -> basename search fallback
    for v in vals:
        p=Path(v)
        candidates=[]
        if p.is_absolute(): candidates.append(p)
        for r in roots: candidates.append(r/p)
        for c in candidates:
            if c.is_file(): return c
    basename=None
    for k in ('event_id','id','uid','name'):
        v=str(item.get(k,''))
        if v:
            basename=v
            break
    if basename:
        pats=[basename, basename+'.npy', basename.replace('/','_')+'.npy']
        for r in roots:
            for pat in pats:
                hits=list(r.rglob(pat)) if r.exists() else []
                if hits: return hits[0]
    return None

def _load_motion(path: Path) -> np.ndarray:
    x=np.load(path,allow_pickle=True)
    if isinstance(x,np.ndarray) and x.ndim==0 and isinstance(x.item(),dict):
        d=x.item(); x=d.get('motion',d.get('pose',d.get('arr_0',x)))
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==3 and x.shape[0]==1: x=x[0]
    if x.ndim!=2 or x.shape[1]<151: raise ValueError(f'{path}: expected [T,151], got {x.shape}')
    return x[:,:151]

def _fk(motion: np.ndarray) -> np.ndarray:
    t=motion.shape[0]; root=motion[:,[4,5,6]]
    local=_rot6d_to_matrix_np(motion[:,7:151].reshape(t,24,6))
    joints=np.zeros((t,24,3),dtype=np.float32); glob=np.zeros((t,24,3,3),dtype=np.float32)
    joints[:,0]=root; glob[:,0]=local[:,0]
    for j in range(1,24):
        p=int(PARENTS[j]); glob[:,j]=np.matmul(glob[:,p],local[:,j])
        joints[:,j]=joints[:,p]+np.matmul(glob[:,p],OFFSETS[j][None,:,None])[...,0]
    return joints

def audit_motion(motion: np.ndarray, quantile: float, margin: float, tolerance: float, weight: float) -> Dict[str, float]:
    joints=_fk(motion); foot_y=joints[:,FOOT_JOINTS,1]
    floor_y=float(np.quantile(foot_y.reshape(-1), quantile))
    min_y=float(np.min(foot_y))
    pen=max(0.0, floor_y+float(margin)-min_y)
    excess=max(0.0, pen-float(tolerance))
    return {
        'native_min_foot_y': min_y,
        'native_floor_y': floor_y,
        'native_floor_penetration_m': float(pen),
        'native_floor_penalty': float(weight*excess*excess),
        'native_floor_ok': bool(pen <= tolerance),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--index_json',required=True)
    ap.add_argument('--out_json',required=True)
    ap.add_argument('--audit_json',default='')
    ap.add_argument('--search_root',action='append',default=[])
    ap.add_argument('--quantile',type=float,default=float(os.getenv('V40_NATIVE_FLOOR_QUANTILE','0.05')))
    ap.add_argument('--margin',type=float,default=float(os.getenv('V40_NATIVE_FLOOR_MARGIN','0.006')))
    ap.add_argument('--tolerance',type=float,default=float(os.getenv('V40_NATIVE_FLOOR_TOLERANCE_M','0.04')))
    ap.add_argument('--penalty_weight',type=float,default=float(os.getenv('V40_NATIVE_FLOOR_PENALTY_WEIGHT','8.0')))
    ap.add_argument('--filter_mode',choices=['none','remove','quality'],default=os.getenv('V40_NATIVE_FLOOR_FILTER_MODE','quality'))
    ap.add_argument('--remove_threshold',type=float,default=float(os.getenv('V40_NATIVE_FLOOR_REMOVE_THRESHOLD','0.16')))
    args=ap.parse_args()
    in_path=Path(args.index_json)
    obj=_load_json(in_path)
    items=_items(obj)
    roots=[Path(p) for p in args.search_root]
    roots += [in_path.parent, Path('.'), Path('data'), Path('output')]
    new_items=[]; rows=[]; missing=0; removed=0; scored=0
    for i,item in enumerate(items):
        row=dict(item)
        mp=_find_motion_path(row, roots)
        stat={'index':i,'motion_path':str(mp) if mp else None}
        if mp is None:
            missing+=1; stat.update({'status':'missing_motion'})
            row['native_floor_available']=False
        else:
            try:
                m=_load_motion(mp); metrics=audit_motion(m,args.quantile,args.margin,args.tolerance,args.penalty_weight)
                row.update(metrics); row['native_floor_available']=True; scored+=1; stat.update(metrics); stat['status']='ok'
                if args.filter_mode=='quality':
                    for key in ('quality_score','quality'):
                        if key in row:
                            try: row[key]=float(row[key])-float(metrics['native_floor_penalty'])
                            except Exception: pass
                    row['v40_native_floor_quality_adjusted']=True
                if args.filter_mode=='remove' and metrics['native_floor_penetration_m']>args.remove_threshold:
                    removed+=1; stat['removed']=True; rows.append(stat); continue
            except Exception as exc:
                missing+=1; row['native_floor_available']=False; stat.update({'status':'failed','error':repr(exc)})
        rows.append(stat); new_items.append(row)
    if isinstance(obj,list): out_obj=new_items
    else:
        out_obj=dict(obj)
        for key in ('items','events','index','data'):
            if isinstance(out_obj.get(key), list):
                out_obj[key]=new_items; break
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_obj,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'input':str(in_path),'output':str(args.out_json),'num_before':len(items),'num_after':len(new_items),'scored':scored,'missing_or_failed':missing,'removed':removed,'filter_mode':args.filter_mode,'tolerance':args.tolerance,'remove_threshold':args.remove_threshold}
    if rows:
        vals=[r.get('native_floor_penetration_m',0.0) for r in rows if 'native_floor_penetration_m' in r]
        if vals:
            summary.update({'max_native_floor_penetration_m':float(max(vals)),'mean_native_floor_penetration_m':float(np.mean(vals)),'num_over_tolerance':int(sum(v>args.tolerance for v in vals)),'num_over_remove_threshold':int(sum(v>args.remove_threshold for v in vals))})
    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True,exist_ok=True)
        Path(args.audit_json).write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
