"""No-op compatibility shim.

The stage-4 / stage-5 trajectory control is now implemented directly in
model/model.py.  This file intentionally does not monkey-patch anything.

It is kept because train.py and generate_controlled.py import
install_native_trajectory_control_patch() at startup in the current branch.
"""

def install_native_trajectory_control_patch(verbose=True):
    if verbose:
        print("ℹ️ trajectory_native_control.py is now a no-op; direct stage-4/5 model is active.")
    return True


def install():
    return install_native_trajectory_control_patch()
