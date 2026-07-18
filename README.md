# V46.53.1 Event-DB compatibility hotfix

The V46.53.1 retargeter correctly changed the root-orientation contract from the
legacy `absolute_reference_lock` to `soft_geodesic_anchor`. The preserved
V46.50 Event-DB builder still accepted only the legacy string and therefore
rejected all otherwise valid V46.53.1 source caches.

This hotfix changes only the Event-DB cache validator. It accepts:

- legacy `absolute_reference_lock`; or
- V46.53.1 `soft_geodesic_anchor` **only when** the report schema, contract
  version, source gate, anatomy gate, gravity gate and fit gate are all valid.

It does not disable strict cache checking and does not weaken the source safety
contract.

## Install

```bash
bash install_hotfix.sh /home/disk/lsm/storage/EDGE
```

## Resume from the successful 12/12 retarget cache

```bash
bash resume_after_retarget.sh
```

The default resume run reuses:

```text
output/v46_53_1_research_20260718_175759/retarget_cache
```

and rebuilds the Event-DB, Grounder, V44, V45, V46, schedule, whole-song motion,
audits and render.
