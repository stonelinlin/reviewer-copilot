# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangExtract: Extract structured information from text with LLMs.

This package provides the main extract and visualize functions,
with lazy loading for other submodules accessed via attribute access.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict

from reviewer.entity_extract import visualization
from reviewer.entity_extract.extraction import extract as extract_func

__all__ = [
    # Public convenience functions (thin wrappers)
    "extract",
    "visualize",
    # Submodules exposed lazily on attribute access for ergonomics:
    "annotation",
    "data",
    "factory",
    "resolver",
    "prompting",
    "io",
    "visualization",
    "exceptions",
    "core",
]

_CACHE: Dict[str, Any] = {}


def extract(*args: Any, **kwargs: Any):
  """Top-level API: lx.extract(...)."""
  return extract_func(*args, **kwargs)


def visualize(*args: Any, **kwargs: Any):
  """Top-level API: lx.visualize(...)."""
  return visualization.visualize(*args, **kwargs)


# PEP 562 lazy loading
_LAZY_MODULES = {
    "annotation": "reviewer.entity_extract.annotation",
    "chunking": "reviewer.entity_extract.chunking",
    "data_lib": "reviewer.entity_extract.data_lib",
    "debug_utils": "reviewer.entity_extract.core.debug_utils",
    "exceptions": "reviewer.entity_extract.exceptions",
    "factory": "reviewer.entity_extract.factory",
    "io": "reviewer.entity_extract.io",
    "progress": "reviewer.entity_extract.progress",
    "prompting": "reviewer.entity_extract.prompting",
    "providers": "reviewer.entity_extract.providers",
    "resolver": "reviewer.entity_extract.resolver",
    "visualization": "reviewer.entity_extract.visualization",
    "core": "reviewer.entity_extract.core",
}


def __getattr__(name: str) -> Any:
  if name in _CACHE:
    return _CACHE[name]
  modpath = _LAZY_MODULES.get(name)
  if modpath is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module = importlib.import_module(modpath)
  # ensure future 'import entity_extract.<name>' returns the same module
  sys.modules[f"{__name__}.{name}"] = module
  setattr(sys.modules[__name__], name, module)
  _CACHE[name] = module
  return module


def __dir__():
  return sorted(__all__)
