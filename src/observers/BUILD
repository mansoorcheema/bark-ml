py_library(
    name = "observers",
    srcs = ["nearest_state_observer.py",
            "simple_observer.py",
            "graph_observer.py",
            "graph_observer_v2.py",
            "observer.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//src/commons:spaces"],
    visibility = ["//visibility:public"],
)