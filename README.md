# “Due to industrial confidentiality, part of the training pipeline is not publicly available.”

Algorithm 1: Multi-model Double-Sided Defect Inspection Pipeline

Input:
    S = incoming fudge sample
    M = {m1, m2, ..., mk}  // object detectors (e.g., YOLOv4, YOLOv5, YOLOv7, YOLOv8, YOLOv11, SSD)
    W = validation-learned class-wise model weights
    T = temperature parameters for confidence calibration
    R = rule library for class-priority selection
    θconf = confidence threshold
    θnms = class-wise NMS thresholds
    τdef, τnorm = normal-veto thresholds

Output:
    y_final ∈ {hole, leak, white, normal}
    a control command ∈ {reject, pass}
    logged inspection record

Procedure:
1:  Acquire front-side image Ifront from sample S
2:  Run all detectors in M on Ifront in parallel
3:  Collect candidate detections Dfront = {d1, d2, ..., dn}
4:  Remove candidates with confidence < θconf
5:  Apply class-wise NMS to Dfront using θnms
6:  For each remaining detection d from model mi:
7:      Calibrate class confidence by temperature scaling using Ti
8:  End for
9:  Compute class-wise fused scores on front side:
        Sc(front) = Σi wi,c · p̃i,c
10: Apply normal-veto rule if all defect scores < τdef and Snormal ≥ τnorm
11: If near-tied fused scores occur, resolve by:
        (a) class-priority rule library R, then
        (b) majority voting across detectors
12: Obtain fused front-side decision yfront

13: If yfront ≠ normal then
14:     Issue reject command
15:     Log front-side result and terminate
16: End if

17: Flip sample S mechanically
18: Acquire back-side image Iback
19: Run all detectors in M on Iback in parallel
20: Collect candidate detections Dback
21: Remove candidates with confidence < θconf
22: Apply class-wise NMS to Dback
23: Calibrate class confidence by temperature scaling
24: Compute back-side fused scores:
        Sc(back) = Σi wi,c · p̃i,c
25: Apply normal-veto and tie-resolution rules as in Steps 10–11
26: Obtain fused back-side decision yback

27: Combine two-sided decisions by union rule:
        if yfront or yback is a defect class,
            y_final = defect class with stronger fused confidence
        else
            y_final = normal

28: If y_final = normal then
29:     Issue pass command
30: Else
31:     Issue reject command
32: End if

33: Log front-side result, back-side result, final decision, and metadata
34: Return y_final
