# Human Evaluation Template for EDGE / ChoreoRAG

## Goal
Blindly compare Baseline / No-RAG / No-Context outputs with ChoreoRAG V10 outputs.
Use shuffled A/B order and hide method names from raters.

## Recommended conditions
- A: clean V9 baseline or V10 no_context
- B: V10 ChoreoRAG with Beam Search + nonlinear transition penalty + Temporal Unit Prior
- Optional C: V10 shuffled_context / wrong_text ablation

## Rating scale
1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent.

## Questions
1. Motion Richness: Are upper-body motions expressive, varied, and compatible with Dunhuang/Flying-Apsaras style?
2. Transition Naturalness: Are transitions free of non-human jumps, hard cuts, or high-jerk twitching?
3. Locomotion Quality: Is lower-body/root movement natural, without obvious skating, moonwalk, or floating?
4. Dunhuang Style Consistency: Does the motion look like Dunhuang / Feitian / classical dance rather than generic dance?
5. Overall Preference: Which video is better overall?

## Optional pairwise record format
| case_id | method_A_hidden | method_B_hidden | richness_A | richness_B | transition_A | transition_B | locomotion_A | locomotion_B | style_A | style_B | preference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
