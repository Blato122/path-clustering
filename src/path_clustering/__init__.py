__all__ = [
    "ClusteredRouteOutputs",
    "generate_clustered_routes",
    "generate_named_route_set",
]


def __getattr__(name):
    if name in __all__:
        from .pipeline import (
            ClusteredRouteOutputs,
            generate_clustered_routes,
            generate_named_route_set,
        )

        exports = {
            "ClusteredRouteOutputs": ClusteredRouteOutputs,
            "generate_clustered_routes": generate_clustered_routes,
            "generate_named_route_set": generate_named_route_set,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
