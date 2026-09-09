# Upstream compatibility notice

The `upstream_first` behavior is tested against:

- ZJUTongYang/NUC
- commit `f28c9a0b182d3e7b6b6223972ce7682e7e3b1300`
- upstream license: GNU General Public License v3.0

The upstream source is not modified or vendored here. `upstream.py` loads a separately built
upstream extension for regression testing. `adapter.py` provides the experiment's controlled
facet-expansion policies and records every expansion decision.
