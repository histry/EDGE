#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V38 contact-denoise + SmoothStep foot-lock + targeted Butterworth filter.

Drop-in replacement for tools/v34_motion_quality_postprocess.py.
Keeps old CLI arguments and adds V38_* switches.  The goal is not global
smoothing; it stabilizes contact labels before IK/root locking, ramps contact
lock weights with smoothstep, and low-pass filters only IK-sensitive rotations.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
except Exception:
    torch = None
try:
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton
except Exception:
    ax_from_6v = None; SMPLSkeleton = None
try:
    from scipy.signal import butter, filtfilt
except Exception:
    butter = None; filtfilt = None

PARENTS=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],dtype=np.int64)
OFFSETS=np.array([[0,0,0],[-.10,-.10,0],[.10,-.10,0],[0,.13,0],[0,-.42,0],[0,-.42,0],[0,.14,0],[0,-.40,0],[0,-.40,0],[0,.14,0],[0,-.08,.12],[0,-.08,.12],[0,.14,0],[-.10,.08,0],[.10,.08,0],[0,.16,0],[-.18,0,0],[.18,0,0],[-.28,0,0],[.28,0,0],[-.25,0,0],[.25,0,0],[-.08,0,0],[.08,0,0]],dtype=np.float32)
FOOT_JOINTS=np.array([7,8,10,11],dtype=np.int64)
UPPER_BODY_JOINTS=np.array([13,14,16,17,18,19,20,21,22,23],dtype=np.int64)
COLLISION_PAIRS=((18,3),(19,3),(20,3),(21,3),(22,3),(23,3),(18,6),(19,6),(20,6),(21,6),(22,6),(23,6),(18,9),(19,9),(20,9),(21,9),(22,9),(23,9),(20,12),(21,12),(22,23),(20,21))
JOINT_NAME_TO_ID={"root":0,"lhip":1,"rhip":2,"spine":3,"lknee":4,"rknee":5,"belly":6,"lankle":7,"rankle":8,"chest":9,"ltoes":10,"rtoes":11,"neck":12,"linshoulder":13,"rinshoulder":14,"head":15,"lshoulder":16,"rshoulder":17,"lelbow":18,"relbow":19,"lwrist":20,"rwrist":21,"lhand":22,"rhand":23}

def _enabled(n,d="1"): return os.getenv(n,d).strip().lower() in {"1","true","yes","on"}
def _ef(n,d):
    try: return float(os.getenv(n,str(d)))
    except Exception: return float(d)
def _ei(n,d):
    try: return int(float(os.getenv(n,str(d))))
    except Exception: return int(d)

def _load_motion(path:Path)->np.ndarray:
    x=np.load(path,allow_pickle=True)
    if isinstance(x,np.ndarray) and x.ndim==0 and isinstance(x.item(),dict):
        o=x.item(); x=o.get("motion",o.get("pose",o.get("arr_0",x)))
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==3 and x.shape[0]==1: x=x[0]
    if x.ndim!=2 or x.shape[1]!=151: raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x

def _rot6d_to_matrix_np(x):
    a1=x[...,0:3]; a2=x[...,3:6]
    b1=a1/np.maximum(np.linalg.norm(a1,axis=-1,keepdims=True),1e-8)
    proj=np.sum(b1*a2,axis=-1,keepdims=True)*b1
    b2=a2-proj; b2=b2/np.maximum(np.linalg.norm(b2,axis=-1,keepdims=True),1e-8)
    b3=np.cross(b1,b2)
    return np.stack([b1,b2,b3],axis=-1).astype(np.float32)

def _normalize_6d(x):
    m=_rot6d_to_matrix_np(x)
    return np.concatenate([m[...,0],m[...,1]],axis=-1).astype(np.float32)

def _fk_from_t151_np(motion):
    if _enabled("V34_POSTPROCESS_USE_SMPL_FK","1") and torch is not None and ax_from_6v is not None and SMPLSkeleton is not None:
        try:
            dev=torch.device(os.getenv("V34_POSTPROCESS_FK_DEVICE","cuda" if torch.cuda.is_available() else "cpu"))
            root=torch.tensor(motion[:,[4,5,6]],dtype=torch.float32,device=dev).unsqueeze(0)
            q=torch.tensor(motion[:,7:151],dtype=torch.float32,device=dev).reshape(1,motion.shape[0],24,6)
            with torch.no_grad():
                joints=SMPLSkeleton(device=dev).forward(ax_from_6v(q),root)
            return joints[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
    t=motion.shape[0]; root=motion[:,[4,5,6]].astype(np.float32)
    local=_rot6d_to_matrix_np(motion[:,7:151].reshape(t,24,6))
    joints=np.zeros((t,24,3),dtype=np.float32); glob=np.zeros((t,24,3,3),dtype=np.float32)
    joints[:,0]=root; glob[:,0]=local[:,0]
    for j in range(1,24):
        p=int(PARENTS[j]); glob[:,j]=np.matmul(glob[:,p],local[:,j])
        joints[:,j]=joints[:,p]+np.matmul(glob[:,p],OFFSETS[j][None,:,None])[...,0]
    return joints

def _moving_average(x,window):
    window=int(window)
    if window<=1 or len(x)<3: return x.astype(np.float32,copy=True)
    if window%2==0: window+=1
    pad=window//2; k=np.ones((window,),dtype=np.float32)/float(window)
    flat=x.reshape(x.shape[0],-1); padded=np.pad(flat,((pad,pad),(0,0)),mode="edge")
    out=np.empty_like(flat,dtype=np.float32)
    for i in range(flat.shape[1]): out[:,i]=np.convolve(padded[:,i],k,mode="valid")
    return out.reshape(x.shape).astype(np.float32)

def _segments(mask,min_len):
    rows=[]; start=None
    for i,flag in enumerate(mask.astype(bool).tolist()+[False]):
        if flag and start is None: start=i
        elif (not flag) and start is not None:
            if i-start>=int(min_len): rows.append((start,i))
            start=None
    return rows

def _remove_short_runs(x,min_run):
    y=np.zeros_like(x,dtype=bool)
    for s,e in _segments(x,1):
        if e-s>=int(min_run): y[s:e]=True
    return y

def _median_bool(x,size):
    size=max(1,int(size))
    if size<=1 or len(x)<3: return x.astype(bool)
    if size%2==0: size+=1
    pad=size//2; y=np.pad(x.astype(np.uint8),(pad,pad),mode="edge")
    out=np.zeros_like(x,dtype=bool); need=size//2+1
    for i in range(len(x)): out[i]=int(np.sum(y[i:i+size]))>=need
    return out

def _fill_short_gaps(x,max_gap):
    y=x.astype(bool).copy()
    if max_gap<=0: return y
    for s,e in _segments(~y,1):
        if s>0 and e<len(y) and y[s-1] and y[e] and (e-s)<=int(max_gap):
            y[s:e]=True
    return y

def _denoise_contact(raw,enabled=True,median_size=5,close_holes=5,open_spikes=3,min_contact_frames=3):
    raw=raw.astype(bool)
    if raw.ndim!=2 or raw.shape[1]!=4: return raw,{"enabled":False,"reason":"bad_shape"}
    if not enabled:
        clean=np.zeros_like(raw,dtype=bool)
        for f in range(4): clean[:,f]=_remove_short_runs(raw[:,f],min_contact_frames)
        return clean,{"enabled":False}
    clean=np.zeros_like(raw,dtype=bool); rows=[]
    for f in range(4):
        x=raw[:,f]
        y=_median_bool(x,median_size)
        y=_fill_short_gaps(y,close_holes)
        y=_remove_short_runs(y,max(open_spikes,min_contact_frames))
        clean[:,f]=y
        rows.append({"foot":f,"before_rate":float(np.mean(x)),"after_rate":float(np.mean(y)),"changed_frames":int(np.sum(x!=y))})
    return clean,{"enabled":True,"median_size":int(median_size),"close_holes":int(close_holes),"open_spikes":int(open_spikes),"min_contact_frames":int(min_contact_frames),"per_foot":rows}

def _contact_mask(motion,threshold,min_contact_frames=1,denoise=True,median_size=5,close_holes=5,open_spikes=3):
    label=np.asarray(motion[:,0:4],dtype=np.float32)>=float(threshold)
    raw=label
    if _enabled("V34_KINEMATIC_CONTACT_INFER","1") and len(motion)>=2:
        try:
            joints=_fk_from_t151_np(motion); feet=joints[:,FOOT_JOINTS,:]; foot_y=feet[:,:,1]
            q=float(np.clip(_ef("V34_FLOOR_QUANTILE",0.12),0.01,0.45))
            floor=float(np.quantile(foot_y.reshape(-1),q))
            height=foot_y<=floor+_ef("V34_KIN_CONTACT_HEIGHT",0.045)
            speed=np.zeros_like(foot_y,dtype=np.float32)
            speed[1:]=np.linalg.norm(feet[1:,:,[0,2]]-feet[:-1,:,[0,2]],axis=-1)
            speed_gate=speed<=_ef("V34_KIN_CONTACT_SPEED",0.035)
            votes=label.astype(np.int32)+height.astype(np.int32)+speed_gate.astype(np.int32)
            raw=votes>=int(np.clip(_ef("V34_KIN_CONTACT_VOTE_THRESHOLD",2.0),1,3))
        except Exception:
            raw=label
    clean,_=_denoise_contact(raw,enabled=denoise,median_size=median_size,close_holes=close_holes,open_spikes=open_spikes,min_contact_frames=max(1,int(min_contact_frames)))
    return clean

def _mean_contact_speed(joints,contact):
    feet=joints[:,FOOT_JOINTS,:]; speed=np.zeros(feet.shape[:2],dtype=np.float32)
    speed[1:]=np.linalg.norm(feet[1:,:,[0,2]]-feet[:-1,:,[0,2]],axis=-1)
    return float(np.mean(speed[contact.astype(bool)])) if np.any(contact) else 0.0

def _collision_stats(joints,radius):
    dists=[np.linalg.norm(joints[:,a]-joints[:,b],axis=-1) for a,b in COLLISION_PAIRS]
    dist=np.stack(dists,axis=1); pen=np.maximum(0.0,float(radius)-dist)
    return {"risk":float(np.mean(np.square(pen/max(float(radius),1e-8)))),"min_distance":float(np.min(dist)),"bad_frames":int(np.sum(np.any(pen>0,axis=1)))}

def _quality_audit(motion,contact_threshold,min_contact_frames,floor_margin,collision_radius,denoise,median_size,close_holes,open_spikes):
    joints=_fk_from_t151_np(motion); feet=joints[:,FOOT_JOINTS,:]; foot_y=feet[:,:,1]
    q=float(np.clip(_ef("V34_FLOOR_QUANTILE",0.12),0.01,0.45))
    floor=float(_ef("V34_FLOOR_Y",float(np.quantile(foot_y.reshape(-1),q))))
    contact=_contact_mask(motion,contact_threshold,min_contact_frames,denoise,median_size,close_holes,open_spikes)
    feet_xz=feet[:,:,[0,2]]; vals=[]
    for f in range(4):
        for s,e in _segments(contact[:,f],max(2,int(min_contact_frames))):
            if e-s>1: vals.append(np.linalg.norm(feet_xz[s+1:e,f]-feet_xz[s:e-1,f],axis=-1))
    skate=float(np.mean(np.concatenate(vals))) if vals else 0.0
    root_y=motion[:,5]; root_v=np.diff(root_y,prepend=root_y[:1]); root_a=np.diff(root_v,prepend=root_v[:1])
    vel=np.diff(joints,axis=0,prepend=joints[:1]); acc=np.diff(vel,axis=0,prepend=vel[:1]); jerk=np.diff(acc,axis=0,prepend=acc[:1])
    mean_jerk=np.linalg.norm(jerk,axis=-1).mean(axis=-1); col=_collision_stats(joints,collision_radius)
    penetration=foot_y-(floor+float(floor_margin))
    return {"floor_y":floor,"contact_ratio":float(np.mean(contact)),"foot_skate_mean_mpf":skate,"foot_penetration_min_m":float(np.min(penetration)),"root_y_range_m":float(np.max(root_y)-np.min(root_y)),"root_y_acc_mean":float(np.mean(np.abs(root_a))),"root_y_acc_p95":float(np.percentile(np.abs(root_a),95)),"mean_joint_jerk_max":float(np.max(mean_jerk)),"mean_joint_jerk_p95":float(np.percentile(mean_jerk,95)),"collision_risk":float(col["risk"]),"collision_min_distance":float(col["min_distance"]),"collision_bad_frames":float(col["bad_frames"])}

def _quality_rejection_signal(m):
    reasons=[]
    if m.get("foot_skate_mean_mpf",0)>_ef("V34_REJECT_MAX_SKATE_MPF",0.035): reasons.append("foot_sliding")
    if m.get("foot_penetration_min_m",0)<_ef("V34_REJECT_MIN_FOOT_PENETRATION",-0.015): reasons.append("floor_penetration")
    if m.get("mean_joint_jerk_p95",0)>_ef("V34_REJECT_MAX_JERK_P95",1200.0): reasons.append("high_jitter")
    if m.get("collision_risk",0)>_ef("V34_REJECT_MAX_COLLISION_RISK",0.20): reasons.append("self_collision_proxy")
    return {"accepted":not reasons,"reject_reasons":reasons,"planner_action":"accept" if not reasons else "reroute_local_phrase_or_mask_failed_edges"}

def _smoothstep(u):
    u=np.clip(u,0,1).astype(np.float32); return (3*u*u-2*u*u*u).astype(np.float32)

def _seg_weights(n,blend):
    w=np.ones((n,),dtype=np.float32); b=max(0,min(int(blend),max(0,n//2)))
    if b>0:
        ramp=_smoothstep((np.arange(b,dtype=np.float32)+1.0)/float(b))
        w[:b]=np.minimum(w[:b],ramp); w[-b:]=np.minimum(w[-b:],ramp[::-1])
    return w

def contact_lock_root(motion,contact_threshold,min_contact_frames,strength,smooth_window,max_correction,denoise=True,median_size=5,close_holes=5,open_spikes=3,blend_frames=5):
    out=motion.astype(np.float32,copy=True); joints=_fk_from_t151_np(out)
    raw=_contact_mask(out,contact_threshold,1,False,median_size,close_holes,open_spikes)
    contact,den_summary=_denoise_contact(raw,denoise,median_size,close_holes,open_spikes,min_contact_frames)
    corr=np.zeros((len(out),2),dtype=np.float32); weight=np.zeros((len(out),1),dtype=np.float32); feet=joints[:,FOOT_JOINTS,:]; segs=0
    for f in range(4):
        for s,e in _segments(contact[:,f],min_contact_frames):
            seg=feet[s:e,f,:][:,[0,2]]; anchor=np.median(seg,axis=0)
            w=_seg_weights(e-s,blend_frames)[:,None]
            corr[s:e]+=w*(anchor[None,:]-seg); weight[s:e]+=w; segs+=1
    active=weight[:,0]>1e-6; corr[active]/=np.maximum(weight[active],1e-6)
    if max_correction>0:
        n=np.linalg.norm(corr,axis=1,keepdims=True); corr*=np.minimum(1.0,float(max_correction)/np.maximum(n,1e-8))
    corr=_moving_average(corr,smooth_window); lw=np.clip(weight,0,1)[:,0]
    out[:,4]+=float(strength)*lw*corr[:,0]; out[:,6]+=float(strength)*lw*corr[:,1]
    after=_fk_from_t151_np(out)
    return out,{"enabled":True,"version":"v38_smoothstep_contact_lock","contact_segments":int(segs),"active_contact_frames":int(np.sum(active)),"mean_contact_speed_before":_mean_contact_speed(joints,contact),"mean_contact_speed_after":_mean_contact_speed(after,contact),"max_root_xz_correction":float(np.max(np.linalg.norm(corr,axis=1))) if len(out) else 0.0,"strength":float(strength),"smoothstep_blend_frames":int(blend_frames),"denoise":den_summary}

def enforce_root_y_physics(motion,contact_threshold,min_flight_frames,parabola_strength,min_arc_lift,max_arc_lift,landing_frames,landing_max_drop,landing_strength,denoise=True,median_size=5,close_holes=5,open_spikes=3):
    out=motion.astype(np.float32,copy=True); contact=_contact_mask(out,contact_threshold,max(2,_ei("V34_KIN_CONTACT_MIN_FRAMES",3)),denoise,median_size,close_holes,open_spikes)
    support=np.any(contact,axis=1); flights=_segments(~support,min_flight_frames)
    y=out[:,5].copy(); corrected=y.copy()
    for s,e in flights:
        n=e-s
        if n<2: continue
        u=np.linspace(0,1,n,dtype=np.float32); y0=float(y[s]); y1=float(y[e-1])
        lin=(1-u)*y0+u*y1; obs=float(np.max(y[s:e])-max(y0,y1)); lift=float(np.clip(max(obs,min_arc_lift),0,max_arc_lift))
        corrected[s:e]=(1-float(parabola_strength))*corrected[s:e]+float(parabola_strength)*(lin+lift*4*u*(1-u))
    land=0
    if landing_frames>2 and landing_strength>0:
        trans=np.flatnonzero((support[1:]==True)&(support[:-1]==False))+1
        for t in trans:
            e=min(len(out),t+int(landing_frames)); n=e-t
            if n<3: continue
            pre=max(0,t-3); ds=max(0.0,float(np.mean(y[pre:t])-y[t])); amp=float(np.clip(.35*ds+.012,0,landing_max_drop))
            u=np.linspace(0,1,n,dtype=np.float32); corrected[t:e]+=float(landing_strength)*(-amp*np.sin(np.pi*u)*np.exp(-1.35*u)); land+=1
    out[:,5]=corrected.astype(np.float32)
    return out,{"enabled":True,"flight_segments":len(flights),"landing_events":land,"root_y_delta_mean":float(np.mean(np.abs(corrected-y))),"root_y_delta_max":float(np.max(np.abs(corrected-y)))}

def enforce_floor_clearance(motion,enabled,margin,strength,smooth_window,max_lift,contact_threshold,min_contact_frames,support_damping,denoise=True,median_size=5,close_holes=5,open_spikes=3):
    joints=_fk_from_t151_np(motion); feet_y=joints[:,FOOT_JOINTS,1]; q=float(np.clip(_ef("V34_FLOOR_QUANTILE",0.12),0.01,0.45))
    floor=float(_ef("V34_FLOOR_Y",float(np.quantile(feet_y.reshape(-1),q))))
    pen=np.maximum(0.0,floor+float(margin)-np.min(feet_y,axis=1))
    before={"floor_y":floor,"penetrating_frames":int(np.sum(pen>1e-6)),"max_penetration":float(np.max(pen)),"mean_penetration":float(np.mean(pen))}
    if not enabled: return motion.astype(np.float32,copy=True),{"enabled":False,"before":before}
    lift=pen.astype(np.float32)
    if max_lift>0: lift=np.minimum(lift,float(max_lift))
    lift=_moving_average(lift[:,None],smooth_window)[:,0]
    out=motion.astype(np.float32,copy=True); out[:,5]+=float(strength)*lift
    support=np.zeros((len(out),),dtype=bool)
    if support_damping>0 and len(out)>2:
        contact=_contact_mask(out,contact_threshold,min_contact_frames,denoise,median_size,close_holes,open_spikes); support=np.any(contact,axis=1)
        y=out[:,5].copy(); vel=np.zeros_like(y); vel[1:]=y[1:]-y[:-1]; dvel=vel.copy(); dvel[support]*=max(0,1-float(support_damping))
        yd=y.copy(); yd[1:]=y[0]+np.cumsum(dvel[1:]); out[:,5]=(0.65*y+0.35*yd).astype(np.float32)
    aj=_fk_from_t151_np(out); ap=np.maximum(0.0,floor+float(margin)-np.min(aj[:,FOOT_JOINTS,1],axis=1))
    return out,{"enabled":True,"margin":float(margin),"support_damping":float(support_damping),"support_frame_ratio":float(np.mean(support)),"before":before,"after":{"penetrating_frames":int(np.sum(ap>1e-6)),"max_penetration":float(np.max(ap)),"mean_penetration":float(np.mean(ap))},"max_root_y_lift":float(np.max(lift)) if len(lift) else 0.0}

def smooth_rotations_only(motion,rotation_window,strength):
    out=motion.astype(np.float32,copy=True); strength=float(np.clip(strength,0,1))
    if rotation_window>1 and strength>0:
        sm=_moving_average(out[:,7:151],rotation_window); out[:,7:151]=(1-strength)*out[:,7:151]+strength*sm
        out[:,7:151]=_normalize_6d(out[:,7:151].reshape(-1,24,6)).reshape(-1,144)
    return out

def _rot6d_to_matrix_torch(x):
    a1=x[...,0:3]; a2=x[...,3:6]
    b1=a1/torch.clamp(torch.linalg.norm(a1,dim=-1,keepdim=True),min=1e-8)
    proj=torch.sum(b1*a2,dim=-1,keepdim=True)*b1
    b2=a2-proj; b2=b2/torch.clamp(torch.linalg.norm(b2,dim=-1,keepdim=True),min=1e-8)
    b3=torch.cross(b1,b2,dim=-1); return torch.stack([b1,b2,b3],dim=-1)

def _fk_torch(motion):
    t=motion.shape[0]; root=motion[:,[4,5,6]]; local=_rot6d_to_matrix_torch(motion[:,7:151].reshape(t,24,6))
    joints=[]; glob=[]; offsets=torch.as_tensor(OFFSETS,dtype=motion.dtype,device=motion.device)
    for j in range(24):
        if j==0: joints.append(root); glob.append(local[:,0])
        else:
            p=int(PARENTS[j]); glob.append(torch.matmul(glob[p],local[:,j]))
            joints.append(joints[p]+torch.matmul(glob[p],offsets[j].view(1,3,1)).squeeze(-1))
    return torch.stack(joints,dim=1)

def collision_aware_ik(motion,enabled,radius,steps,lr,collision_weight,reg_weight,temporal_weight,device):
    before=_collision_stats(_fk_from_t151_np(motion),radius)
    if not enabled or torch is None or before["bad_frames"]==0 or steps<=0:
        before.update({"enabled":bool(enabled and torch is not None),"skipped":True}); return motion.astype(np.float32,copy=True),before
    dev=torch.device(device if device=="cuda" and torch.cuda.is_available() else "cpu")
    base=torch.as_tensor(motion.astype(np.float32),device=dev); upper=torch.as_tensor(UPPER_BODY_JOINTS,dtype=torch.long,device=dev)
    original=base[:,7:151].reshape(-1,24,6)[:,upper].detach(); var=original.clone().detach().requires_grad_(True); opt=torch.optim.Adam([var],lr=float(lr))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True); full=base[:,7:151].reshape(-1,24,6).clone(); full[:,upper]=var
        cand=base.clone(); cand[:,7:151]=full.reshape(-1,144); joints=_fk_torch(cand)
        losses=[torch.relu(float(radius)-torch.linalg.norm(joints[:,a]-joints[:,b],dim=-1))**2 for a,b in COLLISION_PAIRS]
        coll=torch.stack(losses,dim=1).mean(); reg=torch.mean((var-original)**2)
        temp=torch.mean((var[1:]-var[:-1])**2) if var.shape[0]>2 else torch.zeros((),dtype=var.dtype,device=var.device)
        loss=float(collision_weight)*coll+float(reg_weight)*reg+float(temporal_weight)*temp
        loss.backward(); opt.step()
    out=motion.astype(np.float32,copy=True); rot=out[:,7:151].reshape(-1,24,6); rot[:,UPPER_BODY_JOINTS]=var.detach().cpu().numpy()
    out[:,7:151]=_normalize_6d(rot).reshape(-1,144); after=_collision_stats(_fk_from_t151_np(out),radius)
    return out,{"enabled":True,"skipped":False,"device":str(dev),"steps":int(steps),"before":before,"after":after}

def _parse_joints(text):
    vals=[]
    for it in str(text or "").replace(";",",").split(","):
        it=it.strip().lower()
        if not it: continue
        vals.append(JOINT_NAME_TO_ID[it] if it in JOINT_NAME_TO_ID else int(float(it)))
    return np.asarray(sorted(set([v for v in vals if 0<=v<24])) or [18,19,20,21,22,23,16,17],dtype=np.int64)

def _fft_lowpass(x,fps,cutoff):
    freq=np.fft.rfftfreq(len(x),d=1.0/max(float(fps),1e-6)); X=np.fft.rfft(x,axis=0); X[freq>float(cutoff)]=0
    return np.fft.irfft(X,n=len(x),axis=0).astype(np.float32)

def butterworth_lowpass_rotations(motion,enabled,fps,cutoff_hz,order,strength,joints_text):
    out=motion.astype(np.float32,copy=True)
    if not enabled or len(out)<8 or strength<=0: return out,{"enabled":bool(enabled),"skipped":True}
    ids=_parse_joints(joints_text); rot=out[:,7:151].reshape(len(out),24,6); target=rot[:,ids,:].reshape(len(out),-1).astype(np.float32)
    nyq=.5*float(fps); cutoff=float(np.clip(cutoff_hz,.1,max(.11,nyq*.95))); method="fft_fallback"; filt=None
    if butter is not None and filtfilt is not None and len(out)>3*(int(order)+1):
        try:
            b,a=butter(int(order),cutoff/nyq,btype="low",analog=False); filt=filtfilt(b,a,target,axis=0).astype(np.float32); method="butterworth_filtfilt"
        except Exception: filt=None
    if filt is None: filt=_fft_lowpass(target,fps,cutoff)
    rot[:,ids,:]=((1-float(strength))*target+float(strength)*filt).reshape(len(out),len(ids),6)
    out[:,7:151]=_normalize_6d(rot).reshape(len(out),144)
    return out,{"enabled":True,"skipped":False,"method":method,"fps":float(fps),"cutoff_hz":cutoff,"order":int(order),"strength":float(strength),"joint_ids":[int(x) for x in ids]}

def process_file(args):
    src=Path(args.motion); out_path=Path(args.out); motion=_load_motion(src); original=motion.copy()
    den=bool(args.contact_denoise)
    qargs=dict(contact_threshold=args.contact_threshold,min_contact_frames=args.min_contact_frames,floor_margin=args.floor_margin,collision_radius=args.collision_radius,denoise=den,median_size=args.contact_median_size,close_holes=args.contact_close_holes,open_spikes=args.contact_open_spikes)
    pre=_quality_audit(original,**qargs)
    physics={"enabled":False}
    if args.root_y_physics:
        motion,physics=enforce_root_y_physics(motion,args.contact_threshold,args.min_flight_frames,args.parabola_strength,args.min_arc_lift,args.max_arc_lift,args.landing_frames,args.landing_max_drop,args.landing_strength,den,args.contact_median_size,args.contact_close_holes,args.contact_open_spikes)
    motion,collision=collision_aware_ik(motion,bool(args.collision_ik),args.collision_radius,args.collision_steps,args.collision_lr,args.collision_weight,args.collision_reg_weight,args.collision_temporal_weight,args.device)
    contact={"enabled":False}
    if args.contact_lock:
        motion,contact=contact_lock_root(motion,args.contact_threshold,args.min_contact_frames,args.contact_lock_strength,args.contact_smooth_window,args.max_root_correction,den,args.contact_median_size,args.contact_close_holes,args.contact_open_spikes,args.contact_lock_blend_frames)
    motion,floor=enforce_floor_clearance(motion,bool(args.floor_clearance),args.floor_margin,args.floor_strength,args.floor_smooth_window,args.floor_max_lift,args.contact_threshold,args.min_contact_frames,args.floor_support_damping,den,args.contact_median_size,args.contact_close_holes,args.contact_open_spikes)
    motion,butter_sum=butterworth_lowpass_rotations(motion,bool(args.butterworth_filter),args.fps,args.butterworth_cutoff_hz,args.butterworth_order,args.butterworth_strength,args.butterworth_joints)
    if args.smooth: motion=smooth_rotations_only(motion,args.rotation_smooth_window,args.smooth_strength)
    out_path.parent.mkdir(parents=True,exist_ok=True); np.save(out_path,motion.astype(np.float32))
    post=_quality_audit(motion,**qargs); reject=_quality_rejection_signal(post)
    summary={"version":"v38_contact_denoise_smoothstep_butterworth_postprocess","input":str(src),"output":str(out_path),"frames":int(len(motion)),"pre_audit":pre,"post_audit":post,"planner_feedback":reject,"audit_improvement":{"foot_skate_mean_delta":float(pre["foot_skate_mean_mpf"]-post["foot_skate_mean_mpf"]),"foot_penetration_min_delta":float(post["foot_penetration_min_m"]-pre["foot_penetration_min_m"]),"jerk_p95_delta":float(pre["mean_joint_jerk_p95"]-post["mean_joint_jerk_p95"]),"collision_risk_delta":float(pre["collision_risk"]-post["collision_risk"])},"root_y_physics":physics,"collision_aware_ik":collision,"contact_lock":contact,"floor_clearance":floor,"butterworth_filter":butter_sum,"rotation_smooth":{"enabled":bool(args.smooth),"note":"global smooth kept optional; V38 uses targeted low-pass"},"root_xz_delta_mean":float(np.mean(np.linalg.norm(motion[:,[4,6]]-original[:,[4,6]],axis=1))) if len(motion) else 0.0,"root_xz_delta_max":float(np.max(np.linalg.norm(motion[:,[4,6]]-original[:,[4,6]],axis=1))) if len(motion) else 0.0,"root_y_delta_mean":float(np.mean(np.abs(motion[:,5]-original[:,5]))) if len(motion) else 0.0,"root_y_delta_max":float(np.max(np.abs(motion[:,5]-original[:,5]))) if len(motion) else 0.0}
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True,exist_ok=True); Path(args.summary_json).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--motion",required=True); p.add_argument("--out",required=True); p.add_argument("--summary_json",default=""); p.add_argument("--device",default="cuda"); p.add_argument("--fps",type=float,default=_ef("V26_FPS",30.0))
    p.add_argument("--contact_threshold",type=float,default=0.65)
    p.add_argument("--root_y_physics",type=int,default=1); p.add_argument("--min_flight_frames",type=int,default=6); p.add_argument("--parabola_strength",type=float,default=0.60); p.add_argument("--min_arc_lift",type=float,default=0.012); p.add_argument("--max_arc_lift",type=float,default=0.10); p.add_argument("--landing_frames",type=int,default=8); p.add_argument("--landing_max_drop",type=float,default=0.035); p.add_argument("--landing_strength",type=float,default=0.75)
    p.add_argument("--collision_ik",type=int,default=1); p.add_argument("--collision_radius",type=float,default=0.16); p.add_argument("--collision_steps",type=int,default=24); p.add_argument("--collision_lr",type=float,default=0.025); p.add_argument("--collision_weight",type=float,default=8.0); p.add_argument("--collision_reg_weight",type=float,default=0.45); p.add_argument("--collision_temporal_weight",type=float,default=0.02)
    p.add_argument("--contact_lock",type=int,default=1); p.add_argument("--min_contact_frames",type=int,default=8); p.add_argument("--contact_lock_strength",type=float,default=0.85); p.add_argument("--contact_smooth_window",type=int,default=11); p.add_argument("--max_root_correction",type=float,default=0.18)
    p.add_argument("--floor_clearance",type=int,default=1); p.add_argument("--floor_margin",type=float,default=0.006); p.add_argument("--floor_strength",type=float,default=0.95); p.add_argument("--floor_smooth_window",type=int,default=5); p.add_argument("--floor_max_lift",type=float,default=0.12); p.add_argument("--floor_support_damping",type=float,default=_ef("V34_FLOOR_SUPPORT_DAMPING",0.25))
    p.add_argument("--smooth",type=int,default=0); p.add_argument("--rotation_smooth_window",type=int,default=3); p.add_argument("--smooth_strength",type=float,default=0.20)
    p.add_argument("--contact_denoise",type=int,default=_ei("V38_CONTACT_DENOISE",1)); p.add_argument("--contact_median_size",type=int,default=_ei("V38_CONTACT_MEDIAN_SIZE",5)); p.add_argument("--contact_close_holes",type=int,default=_ei("V38_CONTACT_CLOSE_HOLES",5)); p.add_argument("--contact_open_spikes",type=int,default=_ei("V38_CONTACT_OPEN_SPIKES",3)); p.add_argument("--contact_lock_blend_frames",type=int,default=_ei("V38_CONTACT_LOCK_BLEND_FRAMES",5))
    p.add_argument("--butterworth_filter",type=int,default=_ei("V38_BUTTERWORTH_FILTER",1)); p.add_argument("--butterworth_cutoff_hz",type=float,default=_ef("V38_BUTTERWORTH_CUTOFF_HZ",4.0)); p.add_argument("--butterworth_order",type=int,default=_ei("V38_BUTTERWORTH_ORDER",2)); p.add_argument("--butterworth_strength",type=float,default=_ef("V38_BUTTERWORTH_STRENGTH",0.85)); p.add_argument("--butterworth_joints",default=os.getenv("V38_BUTTERWORTH_JOINTS","lelbow,relbow,lwrist,rwrist,lhand,rhand,lshoulder,rshoulder"))
    process_file(p.parse_args())
if __name__=="__main__": main()
