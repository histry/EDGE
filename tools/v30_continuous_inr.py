#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for the upgraded continuous INR.

The unsafe high-frequency V30 SIREN implementation has been retired from the
main pipeline. Existing imports are redirected to the V34 regularised C3-boundary, contact-aware implementation.
"""
from tools.v32_contact_inr import (  # noqa: F401
    V32ContactINRSystem,
    V32INRConfig,
    c2_quintic_so3_base,
    c2_zero_envelope,
    c3_zero_envelope,
    config_from_dict,
    config_to_dict,
    continuous_time_features,
    linear_beta_schedule,
    make_c2_transition_np,
    selected_timesteps,
)

# Historical aliases.
V30INRConfig = V32INRConfig
V30ContinuousTransitionSystem = V32ContactINRSystem
so3_hermite_base = c2_quintic_so3_base
