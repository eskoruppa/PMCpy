# REPO_TEMPLATE
Template for repos

Add Submodule
```console
git submodule add git@github.com:eskoruppa/<reponame> path/name
git submodule update --init --recursive
```
Update all repos
```console
git submodule update --recursive --remote
```

Clone with all submodules
```console
git clone --recurse-submodules -j8 git@github.com:eskoruppa/cgStiff.git
```
