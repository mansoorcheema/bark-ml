py_library(
    name = "agents",
    srcs = ["base_agent.py",
            "sac_agent.py",
            "sac_agent_gnn.py",
            "td3_agent.py",
            "ppo_agent.py",
            "ppo_agent_gnn.py",
            "tfa_agent.py"],
    visibility = ["//visibility:public"],
    deps = ["@icml2020//source:gnn_wrapper",
            "@icml2020//source:tfa_networks"]
)