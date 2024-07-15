"""Helper API to manage (create, update, delete) a manifest of Kubernetes resources."""

# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.
import copy
import logging
from typing import Callable, Optional, Tuple

from lightkube import Client
from lightkube.core.resource import NamespacedResource, Resource, api_info

from ..types import (
    LightkubeResourcesList,
    LightkubeResourceType,
    LightkubeResourceTypesSet,
)

# TODO: Vendor anything from chisme
from . import apply_many, delete_many


class KubernetesResourceManager:
    """Helper API to manage (create, update, delete) a manifest of Kubernetes resources."""

    def __init__(
        self,
        labels: Optional[dict],
        resource_types: LightkubeResourceTypesSet,
        lightkube_client: Client,
        logger: Optional[logging.Logger] = None,
    ):
        """Return a KubernetesResourceHandler instance.

        Args:
            logger (logging.Logger): (Optional) A logger to use for logging (so that log messages
                                     emitted here will appear under the caller's log namespace).
                                     If not provided, a default logger will be created.
            labels (dict): A dict of labels to use as a label selector for all resources
                           managed by this KRM.  These will be added to any applied resources at
                           .apply() time and will be used to find existing resources in
                           .get_deployed_resources().
                           Recommended input for this is:
                             labels = {
                              'app.kubernetes.io/name': f"{self.model.app.name}-{self.model.name}",
                              'kubernetes-resource-handler-scope': 'some-user-chosen-scope'
                             }
                           See `get_default_labels` for a helper to generate this label dict.
            resource_types (set): Set of Lightkube Resource objects that define the
                                  types of child resources managed by this KRM. Required so KRM can
                                  query the cluster for existing resources to reconcile.
            lightkube_client (lightkube.Client): Lightkube Client to use for all k8s operations.
                                                 This Client must be instantiated with a
                                                 field_manager, otherwise it cannot be used to
                                                 .apply() resources because the kubernetes server
                                                 side apply patch method requires it. A good option
                                                 for this is to use the application name (eg:
                                                 `self.model.app.name` or
                                                 `self.model.app.name +'_' self.model.name`).
        """
        self.labels = labels
        self.resource_types = resource_types
        self.lightkube_client = lightkube_client
        if logger is None:
            self.log = logging.getLogger(__name__)  # TODO: Give default logger a better name
        else:
            self.log = logger

    def apply(self, resources: LightkubeResourcesList, force: bool = True):
        """Apply the provided Kubernetes resources, adding or modifying these objects.

        This can be invoked to create and/or update resources in the kubernetes cluster using
        Kubernetes server-side-apply.

        If self.labels is set, the labels will be added to all resources before applying them.

        If self.resource_types is set, the a ValueError will be raised if trying to create a
        resource not in the set.

        This function will only add or modify existing objects, it will not delete any resources.
        This includes cases where the manifests have changed over time.  For example:
            * calling `krm.apply([PodA])` results in PodA being created
            * subsequently calling `krm.apply([PodB])` results in PodB created and PodA being left
             unchanged
        To simultaneously create, update, and delete resources, see self.reconcile().

        Args:
            resources: A list of Lightkube Resource objects to apply
            force: *(optional)* Force is going to "force" apply requests. It means user will
                   re-acquire conflicting fields owned by other people.
        """
        self.log.info("Applying resources")
        if self.labels is not None:
            resources = _add_labels_to_resources(resources, self.labels)

        if self.resource_types:
            try:
                _validate_resources(resources, allowed_resource_types=self.resource_types)
            except ValueError as e:
                raise ValueError(
                    "Failed to validate resources before applying them. This likely means we tried"
                    " to create a resource of type not included in `KRM.resource_types`."
                ) from e

        apply_many(
            client=self.lightkube_client,
            objs=resources,
            force=force,
            logger=self.log,
        )

    def delete(self, ignore_missing=True):
        """Delete all resources managed by this KubernetesResourceHandler.

        Requires that self.labels and self.resource_types be set.

        Args:
            ignore_missing: *(optional)* Avoid raising 404 errors on deletion (defaults to True)
        """
        resources_to_delete = self.get_deployed_resources()
        delete_many(self.lightkube_client, resources_to_delete, ignore_missing, self.log)

    def get_deployed_resources(self) -> LightkubeResourcesList:
        """Return a list of all resources deployed by this KubernetesResourceHandler.

        Requires that self.labels and self.resource_types be set.

        This method will:
        * for each resource type included in self.resource_types
          * get all resources of that type in the Kubernetes cluster that match the label selector
            defined in self.labels

        Return: A list of Lightkube Resource objects
        """
        if self.labels is None or len(self.labels) == 0:
            raise ValueError("Cannot get_deployed_resources without a labelset defined")

        if self.resource_types is None or len(self.resource_types) == 0:
            raise ValueError("Cannot get_deployed_resources without one or more resource_types")

        resources = []
        for resource_type in self.resource_types:
            if issubclass(resource_type, NamespacedResource):
                # Get resources from all namespaces
                namespace = "*"
            else:
                # Global resources have no namespace
                namespace = None
            resources.extend(
                self.lightkube_client.list(resource_type, namespace=namespace, labels=self.labels)
            )

        return resources

    def reconcile(self, resources: LightkubeResourcesList, force=False, ignore_missing=True):
        """Reconcile the given resources, removing, updating, or creating objects as required.

        This method will:
        * compute the "existing resources" by, for each resource type in self.resource_types,
          getting all resources currently deployed that match the label selector in self.labels
        * compare the existing resources to the desired resources provided, deleting any resources
          that exist but are not in the desired resource list
        * call .apply() to create any new resources and update any remaining existing ones to the
          desired state

        Args:
            resources: A list of Lightkube Resource objects to apply
            force: *(optional)* Passed to self.apply().  This will force apply over any resources
                   marked as managed by another field manager.
            ignore_missing: *(optional)* Avoid raising 404 errors on deletion (defaults to True)
        """
        desired_resources = resources
        existing_resources = self.get_deployed_resources()

        # Delete any resources that exist but are no longer in scope
        resources_to_delete = _in_left_not_right(
            existing_resources, desired_resources, hasher=_hash_lightkube_resource
        )
        delete_many(self.lightkube_client, resources_to_delete, ignore_missing, self.log)

        # Update remaining resources and create any new ones
        self.apply(resources=resources, force=force)


def create_charm_default_labels(application_name: str, model_name: str, scope: str) -> dict:
    """Return a default label style for the KubernetesResourceHandler label selector."""
    return {
        "app.kubernetes.io/instance": f"{application_name}-{model_name}",
        "kubernetes-resource-handler-scope": scope,
    }


def _add_label_field_to_resource(
    resource: LightkubeResourceType,
) -> LightkubeResourceType:
    """Add a metadata.labels field to a Lightkube resource.

    Works around a bug where sometimes when the labels field is None it is not overwritable.
    Converts the object to a dict, adds the labels field, and then converts it back to the
    """
    as_dict = resource.to_dict()
    as_dict["metadata"]["labels"] = {}
    return resource.from_dict(as_dict)


def _add_labels_to_resources(resources: LightkubeResourcesList, labels: dict):
    """Return a copy of resources where each resource has the given labels added."""
    resources = copy.deepcopy(resources)

    for resource in resources:
        if resource.metadata.labels is None:
            resource.metadata.labels = {}

        # Sometimes there is a bug where this field is not overwritable
        if resource.metadata.labels is None:
            resource = _add_label_field_to_resource(resource)
        resource.metadata.labels.update(labels)
    return resources


def _get_resource_classes_in_manifests(
    resource_list: LightkubeResourcesList,
) -> LightkubeResourceTypesSet:
    """Return a set of the resource classes in a list of resources."""
    return {type(rsc) for rsc in resource_list}


def _hash_lightkube_resource(resource: Resource) -> Tuple[str, str, str, str, str]:
    """Hash a Lightkube Resource by returning a tuple of (group, version, kind, name, namespace).

    For global resources or resources without a namespace specified, namespace will be None.
    """
    resource_info = api_info(resource).resource

    return (
        resource_info.group,
        resource_info.version,
        resource_info.kind,
        resource.metadata.name,
        resource.metadata.namespace,
    )


def _in_left_not_right(left: list, right: list, hasher: Optional[Callable] = None) -> list:
    """Return the items in left that are not right (the Set difference).

    Args:
        left: a list
        right: a list
        hasher: (Optional) a function that hashes the items in left and right to something
                immutable that can be compared.  If omitted, will use hash()

    Return:
        A list of items in left that are not in right, based on the hasher function.
    """
    if hasher is None:
        hasher = hash

    left_as_dict = {hasher(resource): resource for resource in left}
    right_as_dict = {hasher(resource): resource for resource in right}

    keys_in_left_not_right = set(left_as_dict.keys()) - set(right_as_dict.keys())
    return [left_as_dict[k] for k in keys_in_left_not_right]


def _validate_resources(resources, allowed_resource_types: LightkubeResourceTypesSet):
    """Validate that the resources are of a type in the allowed_resource_types set.

    Side effect: raises a ValueError if any resource is not in the allowed_resource_types set.

    Args:
        resources: a list of Lightkube resources to validate
        allowed_resource_types: a set of Lightkube resource classes to validate against
    """
    resource_types = _get_resource_classes_in_manifests(resources)
    for resource_type in resource_types:
        if resource_type not in allowed_resource_types:
            raise ValueError(
                f"Resource type {resource_type} not in allowed resource types"
                f" '{allowed_resource_types}'"
            )
