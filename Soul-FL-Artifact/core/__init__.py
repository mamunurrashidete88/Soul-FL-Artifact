# core package — lazy imports (torch required at call-time, not import-time)

def __getattr__(name):
    _models = {"build_model", "get_flat_params", "set_flat_params", "set_flat_params",
               "compute_gradient_update", "model_size_mb", "get_flat_gradients"}
    _client = {"FederatedClient", "build_clients", "SleeperSybilClient",
               "FreeRiderSybilClient", "LazyHoardClient", "AdaptiveManifoldClient"}
    _agg    = {"AggregationEngine"}
    _server = {"SoulFLServer"}

    if name in _models:
        from core import models as _m; return getattr(_m, name)
    if name in _client:
        from core import client as _m; return getattr(_m, name)
    if name in _agg:
        from core import aggregation as _m; return getattr(_m, name)
    if name in _server:
        from core import server as _m; return getattr(_m, name)
    raise AttributeError(f"module 'core' has no attribute {name!r}")


__all__ = [
    "build_model", "get_flat_params", "set_flat_params",
    "FederatedClient", "build_clients",
    "AggregationEngine", "SoulFLServer",
]
