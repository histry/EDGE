# V46.49.4 Absolute Root Orientation Contract

V46.49.3b removes most net low-frequency yaw drift before target IK. However,
the final target motion still contains much more absolute yaw variation than
the corrected source heading, showing that unconstrained target IK reintroduces
yaw oscillation through root/local-joint ambiguity.

V46.49.4 fixes target root orientation to the corrected source body frame while
continuing to optimize root translation and local rotations of joints 1..23.
