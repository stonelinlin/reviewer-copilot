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

"""Compatibility shim for entity_extract.registry imports.

This module redirects to entity_extract.plugins for backward compatibility.
Will be removed in v2.0.0.
"""

from __future__ import annotations

import warnings

from reviewer.entity_extract import plugins


def __getattr__(name: str):
  """Redirect to plugins module with deprecation warning."""
  warnings.warn(
      "`entity_extract.registry` is deprecated and will be removed in v2.0.0; "
      "use `entity_extract.plugins` instead.",
      FutureWarning,
      stacklevel=2,
  )
  return getattr(plugins, name)
