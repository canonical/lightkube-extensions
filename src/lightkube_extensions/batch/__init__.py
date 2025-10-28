# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Utilities for working on batches of resources using the Lightkube library."""

from ._kubernetes_resource_manager import KubernetesResourceManager, create_charm_default_labels
from ._many import apply_many, delete_many, patch_many

__all__ = [KubernetesResourceManager, apply_many, create_charm_default_labels, delete_many, patch_many]
