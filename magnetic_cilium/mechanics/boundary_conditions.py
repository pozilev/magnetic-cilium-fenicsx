from __future__ import annotations

from magnetic_cilium._compat import legacy_attr


def locate_bottom_and_top_facets(domain, params):
    return legacy_attr("mechanics_model", "locate_bottom_and_top_facets")(domain, params)
