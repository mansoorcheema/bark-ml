load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def load_bark():
  # _maybe(
  #   native.local_repository,
  #   name = "bark_project",
  #   path="/home/hart/Dokumente/2020/bark",
  # )
  _maybe(
    native.local_repository,
    name = "icml2020",
    path="/home/hart/Dokumente/2020/icml2020",
  )
  _maybe(
    native.local_repository,
    name = "gnn_lib",
    path="/home/hart/Dokumente/2020/gcn",
  )
  _maybe(
    git_repository,
    name = "bark_project",
    # branch="master",
    commit = "b4b174ce84a65618b5604ebbbc28d1c760ee0db2",
    remote = "https://github.com/bark-simulator/bark",
  )
