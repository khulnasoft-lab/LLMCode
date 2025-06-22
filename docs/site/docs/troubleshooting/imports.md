---
parent: Troubleshooting
nav_order: 28
---

# Dependency versions

Llmcode expects to be installed with the
correct versions of all of its required dependencies.

If you've been linked to this doc from a GitHub issue, 
or if llmcode is reporting `ImportErrors`
it is likely that your
llmcode install is using incorrect dependencies.


## Avoid package conflicts

If you are using llmcode to work on a python project, sometimes your project will require
specific versions of python packages which conflict with the versions that llmcode
requires.
If this happens, you may see errors like these when running pip installs:

```
llmcode 0.23.0 requires somepackage==X.Y.Z, but you have somepackage U.W.V which is incompatible.
```

## Install with llmcode-install, uv or pipx

If you are having dependency problems you should consider
[installing llmcode using llmcode-install, uv or pipx](/docs/install.html).
This will ensure that llmcode is installed in its own python environment,
with the correct set of dependencies.

## Package managers like Homebrew, AUR, ports

Package managers often install llmcode with the wrong dependencies, leading
to import errors and other problems.

It is recommended to
[install llmcode using llmcode-install, uv or pipx](/docs/install.html).


## Dependency versions matter

Llmcode pins its dependencies and is tested to work with those specific versions.
If you are installing llmcode directly with pip
you should be careful about upgrading or downgrading the python packages that
llmcode uses.

In particular, be careful with the packages with pinned versions 
noted at the end of
[llmcode's requirements.in file](https://github.com/KhulnaSoft/llmcode/blob/main/requirements/requirements.in).
These versions are pinned because llmcode is known not to work with the
latest versions of these libraries.

Also be wary of upgrading `litellm`, as it changes versions frequently
and sometimes introduces bugs or backwards incompatible changes.

## Replit

{% include replit-pipx.md %}
