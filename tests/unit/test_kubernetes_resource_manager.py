# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple
from unittest import mock
from unittest.mock import patch

import pytest
from lightkube.codecs import load_all_yaml
from lightkube.generic_resource import create_global_resource, create_namespaced_resource
from lightkube.models.apps_v1 import StatefulSetSpec, StatefulSetStatus
from lightkube.models.core_v1 import PodTemplateSpec
from lightkube.models.meta_v1 import LabelSelector, ObjectMeta
from lightkube.resources.admissionregistration_v1 import MutatingWebhookConfiguration
from lightkube.resources.apps_v1 import StatefulSet
from lightkube.resources.core_v1 import Pod, Service

from lightkube_extensions.batch import KubernetesResourceManager
from lightkube_extensions.batch._kubernetes_resource_manager import (
    _add_labels_to_resources,
    _get_resource_classes_in_manifests,
    _hash_lightkube_resource,
    _in_left_not_right,
    _validate_resources,
)

data_dir = Path(__file__).parent.joinpath("data")

statefulset_with_replicas = StatefulSet(
    metadata=ObjectMeta(name="has-replicas", namespace="namespace"),
    spec=StatefulSetSpec(
        replicas=3, selector=LabelSelector(), serviceName="", template=PodTemplateSpec()
    ),
    status=StatefulSetStatus(replicas=3, readyReplicas=3),
)

statefulset_missing_replicas = StatefulSet(
    metadata=ObjectMeta(name="missing-replicas", namespace="namespace"),
    spec=StatefulSetSpec(
        replicas=3, selector=LabelSelector(), serviceName="", template=PodTemplateSpec()
    ),
    status=StatefulSetStatus(replicas=1, readyReplicas=1),
)


# TODO: Do we even need this?
@pytest.fixture()
def mocked_lightkube(mocker):
    """Prevents lightkube clients from being created, returning a mock instead."""
    yield mocker.MagicMock()


statefulset_dummy = StatefulSet(
    metadata=ObjectMeta(name="has-replicas", namespace="namespace"),
)


DEFAULT_LABELS = {"label": "default"}


@patch("lightkube_extensions.batch._kubernetes_resource_manager.apply_many")
def test_KubernetesResourceManager_apply(  # noqa N802
    mocked_apply_many,
    mocked_lightkube,  # noqa F811
):
    # Dummy that will be returned from render_manifests.  Content doesn't matter because we also
    # mock away apply_many, which will consume this.
    resources = [Pod(metadata=ObjectMeta(name="pod1", namespace="namespace1"))]

    krm = KubernetesResourceManager(
        labels=DEFAULT_LABELS,
        resource_types={Pod},
        lightkube_client=mock.MagicMock(),
    )

    # Act
    krm.apply(resources)

    # Assert that we've tried to apply the resources expected, including the labels added by the KRM
    mocked_apply_many.assert_called_once()
    resources_actual = mocked_apply_many.call_args.kwargs["objs"]
    assert len(resources_actual) == 1
    for k, v in DEFAULT_LABELS.items():
        assert resources_actual[0].metadata.labels[k] == v


def test_KubernetesResourceManager_delete():  # noqa: N802
    """Tests that KRH.delete successfully deletes observed resources."""
    # Arrange
    krm = KubernetesResourceManager(
        labels=DEFAULT_LABELS,
        resource_types={Pod},
        lightkube_client=mock.MagicMock(),
    )

    resources_to_delete = [
        Pod(metadata=ObjectMeta(name="pod1", namespace="namespace1")),
        Service(metadata=ObjectMeta(name="service1", namespace="namespace2")),
    ]
    krm.get_deployed_resources = mock.MagicMock(return_value=resources_to_delete)

    # Act
    krm.delete()

    # Assert
    assert krm.lightkube_client.delete.call_count == len(resources_to_delete)


@pytest.mark.parametrize(
    "labels, resource_types, expected_context",
    [
        (None, {"some", "resources"}, pytest.raises(ValueError)),
        ({}, {"some", "resources"}, pytest.raises(ValueError)),
        ({"some": "labels"}, None, pytest.raises(ValueError)),
        ({"some": "labels"}, {}, pytest.raises(ValueError)),
    ],
)
def test_KubernetesResourceManager_get_deployed_resources_missing_required_arguments(  # noqa: N802
    labels, resource_types, expected_context
):
    """Tests that KRH delete raises when missing required inputs."""
    # Arrange
    krm = KubernetesResourceManager(
        labels=labels,
        resource_types=resource_types,
        lightkube_client=mock.MagicMock(),
    )

    # Act and Assert
    with expected_context:
        krm.get_deployed_resources()


def test_KubernetesResourceManager_get_deployed_resources():  # noqa: N802
    """Tests that KRH.get_deployed_resources returns as expected."""
    # Arrange
    labels = {"some": "labels"}
    resource_types = {Pod, Service}

    def lightkube_client_list_side_effect(resource_type, *args, **kwargs):
        """Returns a list of a single Pod or Service resource, or errors for any other type."""
        if resource_type == Pod:
            return [Pod(metadata=ObjectMeta(name="pod1", namespace="namespace1"))]
        elif resource_type == Service:
            return [Service(metadata=ObjectMeta(name="service1", namespace="namespace2"))]
        else:
            raise ValueError(f"Unexpected resource type: {resource_type}")

    expected_resources = [
        Pod(metadata=ObjectMeta(name="pod1", namespace="namespace1")),
        Service(metadata=ObjectMeta(name="service1", namespace="namespace2")),
    ]

    mock_lightkube_client = mock.MagicMock()
    mock_lightkube_client.list.side_effect = lightkube_client_list_side_effect
    krm = KubernetesResourceManager(
        labels=labels,
        resource_types=resource_types,
        lightkube_client=mock_lightkube_client,
    )

    # Act
    resources = krm.get_deployed_resources()

    # Assert list called once for each resource type
    mock_lightkube_client.list.call_count == len(resource_types)

    # Assert we got the results from list
    assert resources == expected_resources


def test_KubernetesResourceManager_reconcile():  # noqa: N802
    """Test that KRM.reconcile works when expected to."""
    # Arrange
    krm = KubernetesResourceManager(
        labels=DEFAULT_LABELS,
        resource_types={Pod, Service},
        lightkube_client=mock.MagicMock(),
    )

    pods = load_all_yaml((data_dir / "pods_with_labels.j2").read_text())
    services = load_all_yaml((data_dir / "services_with_labels.j2").read_text())

    # desired resources are missing the last service, so we expect it to be deleted
    existing_resources = pods + services
    desired_resources = pods + services[:-1]

    krm.get_deployed_resources = mock.MagicMock(return_value=existing_resources)

    # Act
    krm.reconcile(desired_resources)

    # Assert that we tried to delete one object and then called apply
    assert krm.lightkube_client.delete.call_count == 1
    assert krm.lightkube_client.apply.call_count == len(desired_resources)


@pytest.mark.parametrize(
    "labels, resource_types, expected_context",
    [
        (None, {"some", "resources"}, pytest.raises(ValueError)),
        ({}, {"some", "resources"}, pytest.raises(ValueError)),
        ({"some": "labels"}, None, pytest.raises(ValueError)),
        ({"some": "labels"}, {}, pytest.raises(ValueError)),
    ],
)
def test_KubernetesResourceManager_reconcile_missing_required_arguments(  # noqa: N802
    labels, resource_types, expected_context
):
    """Tests that KRH delete raises when missing required inputs."""
    # Arrange
    krm = KubernetesResourceManager(
        labels=labels,
        resource_types=resource_types,
        lightkube_client=mock.MagicMock(),
    )

    # Act and Assert
    with expected_context:
        krm.reconcile([])


test_namespaced_resource = create_namespaced_resource("mygroup", "myversion", "myres", "myress")
test_global_resource = create_global_resource("mygroup2", "myversion2", "myres2", "myres2s")


@pytest.mark.parametrize(
    "resource, expected",
    [
        (statefulset_with_replicas, ("apps", "v1", "StatefulSet", "has-replicas", "namespace")),
        (
            MutatingWebhookConfiguration(metadata=ObjectMeta(name="name1")),
            ("admissionregistration.k8s.io", "v1", "MutatingWebhookConfiguration", "name1", None),
        ),
        (
            test_namespaced_resource(metadata=ObjectMeta(name="name1", namespace="namespace1")),
            ("mygroup", "myversion", "myres", "name1", "namespace1"),
        ),
        (
            test_global_resource(metadata=ObjectMeta(name="name")),
            ("mygroup2", "myversion2", "myres2", "name", None),
        ),
    ],
)
def test_hash_lightkube_resource(resource, expected):
    """Tests that _hash_lightkube_resource works on a variety of resources."""
    actual = _hash_lightkube_resource(resource)
    assert actual == expected


# Helper that works like a class with an attribute
sample_classlike = NamedTuple("classlike", [("a", int)])


@pytest.mark.parametrize(
    "left, right, hasher, expected",
    [
        ([1, "two", (3, 3, 3)], ["two", (3, 3, 3), 4], None, [1]),
        (
            [sample_classlike(a=1), sample_classlike(a=2)],
            [sample_classlike(a=3), sample_classlike(a=2)],
            lambda x: x.a,
            [sample_classlike(a=1)],
        ),
        (
            [statefulset_with_replicas, statefulset_missing_replicas],
            [statefulset_missing_replicas],
            _hash_lightkube_resource,
            [statefulset_with_replicas],
        ),
        (
            [
                test_global_resource(metadata=ObjectMeta(name="name1")),
                test_global_resource(metadata=ObjectMeta(name="name2")),
            ],
            [
                test_global_resource(metadata=ObjectMeta(name="name2")),
                test_global_resource(metadata=ObjectMeta(name="name3")),
            ],
            _hash_lightkube_resource,
            [test_global_resource(metadata=ObjectMeta(name="name1"))],
        ),
    ],
)
def test_in_left_not_right(left, right, hasher, expected):
    """Tests that _in_left_not_right works on a variety of inputs."""
    actual = _in_left_not_right(left, right, hasher)
    assert actual == expected


@pytest.mark.parametrize(
    "resources, labels, expected",
    [
        ([], {}, []),
        (
            [
                Service(metadata=ObjectMeta(name="name", namespace="namespace")),
                StatefulSet(
                    metadata=ObjectMeta(
                        name="name", namespace="namespace", labels={"starting": "label"}
                    )
                ),
            ],
            {"new": "label!", "anothernew": "label!!"},
            [
                Service(
                    metadata=ObjectMeta(
                        name="name",
                        namespace="namespace",
                        labels={"new": "label!", "anothernew": "label!!"},
                    ),
                ),
                StatefulSet(
                    metadata=ObjectMeta(
                        name="name",
                        namespace="namespace",
                        labels={"starting": "label", "new": "label!", "anothernew": "label!!"},
                    )
                ),
            ],
        ),
    ],
)
def test_add_labels_to_manifest(resources, labels, expected):
    """Tests that _add_labels_to_resources works on a variety of inputs."""
    actual = _add_labels_to_resources(resources, labels)
    assert actual == expected


@pytest.mark.parametrize(
    "resources, expected_classes",
    [
        (
            [
                Service(metadata=ObjectMeta(name="name", namespace="namespace")),
                Service(metadata=ObjectMeta(name="name2", namespace="namespace")),
                StatefulSet(metadata=ObjectMeta(name="name", namespace="namespace")),
                test_global_resource(metadata=ObjectMeta(name="name")),
            ],
            {Service, StatefulSet, test_global_resource},
        ),
    ],
)
def test_get_resource_classes_in_manifests(resources, expected_classes):
    """Tests that _get_resource_classes_in_manifests works with global and namespaced resources."""
    actual_classes = _get_resource_classes_in_manifests(resources)
    assert actual_classes == expected_classes


@pytest.mark.parametrize(
    "resources, allowed_resource_types, expected_context_raised",
    [
        (
            (statefulset_with_replicas, Pod(), test_namespaced_resource()),
            (StatefulSet, Pod, test_namespaced_resource),
            nullcontext(),
        ),
        (
            (statefulset_with_replicas, Pod(), test_namespaced_resource()),
            (StatefulSet, test_namespaced_resource),
            pytest.raises(ValueError),
        ),
    ],
)
def test_validate_resources(resources, allowed_resource_types, expected_context_raised):
    """Tests that _validate_resources correctly validates the resources against an allowed list."""
    with expected_context_raised:
        _validate_resources(resources, allowed_resource_types)
