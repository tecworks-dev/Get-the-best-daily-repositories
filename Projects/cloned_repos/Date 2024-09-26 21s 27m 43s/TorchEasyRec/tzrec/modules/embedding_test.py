# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import unittest
from collections import OrderedDict
from typing import Dict

import torch
from parameterized import parameterized
from torch import nn
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.modules.embedding import (
    EmbeddingGroup,
    EmbeddingGroupImpl,
    SequenceEmbeddingGroupImpl,
)
from tzrec.protos import feature_pb2, model_pb2, module_pb2, seq_encoder_pb2
from tzrec.utils.test_util import TestGraphType, create_test_module


def _create_test_features():
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a", embedding_dim=16, num_buckets=100
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b", embedding_dim=8, num_buckets=1000
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_c", embedding_dim=12, num_buckets=1000
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
        ),
    ]
    features = create_features(feature_cfgs)
    return features


def _create_test_sequence_features():
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a",
                embedding_dim=16,
                num_buckets=100,
                expression="item:cat_a",
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b",
                embedding_dim=8,
                num_buckets=1000,
                expression="item:cat_b",
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="int_a", expression="item:int_a"
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="click_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_a",
                            expression="item:cat_a",
                            embedding_dim=16,
                            num_buckets=100,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_b",
                            expression="item:cat_b",
                            embedding_dim=8,
                            num_buckets=1000,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="int_a",
                            expression="item:int_a",
                        )
                    ),
                ],
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="buy_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_a",
                            expression="item:cat_a",
                            embedding_dim=16,
                            num_buckets=100,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="int_a",
                            expression="item:int_a",
                        )
                    ),
                ],
            )
        ),
    ]
    features = create_features(feature_cfgs)
    return features


EMPTY_KJT = KeyedJaggedTensor.empty()
EMPTY_KT = KeyedTensor(
    keys=[], length_per_key=[], values=torch.empty(0, dtype=torch.float32)
)


class _EGScriptWrapper(nn.Module):
    """Embedding Group inference wrapper for jit.script."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        sparse_features: Dict[str, KeyedJaggedTensor],
        dense_features: Dict[str, KeyedTensor],
        sequence_dense_features: Dict[str, JaggedTensor],
        batch_size: int = -1,
    ):
        return self._module(
            Batch(
                sparse_features=sparse_features,
                dense_features=dense_features,
                sequence_dense_features=sequence_dense_features,
                batch_size=batch_size,
            )
        )


class EmbeddingGroupTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("INPUT_TILE", None)

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_embedding_group_impl(self, graph_type) -> None:
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="wide",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.WIDE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="click",
                        feature_names=[
                            "cat_a",
                            "cat_b",
                            "int_a",
                            "click_seq__cat_a",
                            "click_seq__cat_b",
                            "click_seq__int_a",
                        ],
                    ),
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                ],
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("wide"), [4, 4])
        self.assertEqual(embedding_group.group_dims("deep"), [16, 8, 1])
        self.assertEqual(embedding_group.group_total_dim("wide"), 8)
        self.assertEqual(embedding_group.group_total_dim("deep"), 25)
        wide_feature_dims = OrderedDict({"cat_a": 4, "cat_b": 4})
        deep_feature_dims = OrderedDict({"cat_a": 16, "cat_b": 8, "int_a": 1})
        self.assertDictEqual(
            embedding_group.group_feature_dims("wide"), wide_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("deep"), deep_feature_dims
        )

        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(24))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        result = embedding_group(
            sparse_feature,
            dense_feature,
            EMPTY_KJT,
            EMPTY_KT,
        )

        self.assertEqual(result["wide"].size(), (2, 8))
        self.assertEqual(result["deep"].size(), (2, 25))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_sequence_embedding_group_impl(self, graph_type) -> None:
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_all",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_other",
                feature_names=[
                    "cat_a",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__int_a",
                ],
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_no_query",
                feature_names=["click_seq__cat_a", "click_seq__int_a"],
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(
            embedding_group.group_dims("deep___click_all.sequence"), [16, 8, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_all.query"), [16, 8, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_other.sequence"), [16, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_other.query"), [16, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_no_query.sequence"), [16, 1]
        )
        self.assertEqual(embedding_group.group_dims("deep___click_no_query.query"), [])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_all.sequence"), 25
        )
        self.assertEqual(embedding_group.group_total_dim("deep___click_all.query"), 25)
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_other.sequence"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_other.query"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_no_query.sequence"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_no_query.query"), 0
        )

        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(24))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(10)], dtype=torch.float32),
            lengths=torch.tensor([3, 3, 2, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            EMPTY_KJT,
            EMPTY_KT,
        )
        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))
        self.assertEqual(result["deep___click_all.query"].size(), (2, 25))
        self.assertEqual(result["deep___click_all.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["deep___click_all.sequence_length"].size(), (2,))
        self.assertEqual(result["deep___click_other.query"].size(), (2, 17))
        self.assertEqual(result["deep___click_other.sequence"].size(), (2, 3, 17))
        self.assertEqual(result["deep___click_other.sequence_length"].size(), (2,))
        self.assertTrue("deep___click_no_query.query" not in result)
        self.assertEqual(result["deep___click_no_query.sequence"].size(), (2, 3, 17))
        self.assertEqual(result["deep___click_no_query.sequence_length"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_embedding_group(self, graph_type) -> None:
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="wide",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.WIDE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="click_seq",
                        feature_names=[
                            "cat_a",
                            "cat_b",
                            "int_a",
                            "click_seq__cat_a",
                            "click_seq__cat_b",
                            "click_seq__int_a",
                        ],
                    ),
                    model_pb2.SeqGroupConfig(
                        group_name="buy_seq",
                        feature_names=[
                            "cat_a",
                            "int_a",
                            "buy_seq__cat_a",
                            "buy_seq__int_a",
                        ],
                    ),
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                    seq_encoder_pb2.SeqEncoderConfig(
                        simple_attention=seq_encoder_pb2.SimpleAttention(
                            input="buy_seq"
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="only_buy",
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="buy_seq",
                        feature_names=[
                            "cat_a",
                            "int_a",
                            "buy_seq__cat_a",
                            "buy_seq__int_a",
                        ],
                    )
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        simple_attention=seq_encoder_pb2.SimpleAttention(
                            input="buy_seq"
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = EmbeddingGroup(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("wide"), [4, 4])
        self.assertEqual(embedding_group.group_dims("deep"), [16, 8, 1, 25, 17])
        self.assertEqual(embedding_group.group_dims("only_buy"), [17])
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("wide"), 8)
        self.assertEqual(embedding_group.group_total_dim("deep"), 67)
        self.assertEqual(embedding_group.group_total_dim("only_buy"), 17)
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)
        wide_feature_dims = {"cat_a": 4, "cat_b": 4}
        deep_feature_dims = {
            "cat_a": 16,
            "cat_b": 8,
            "int_a": 1,
            "deep_seq_encoder_0": 25,
            "deep_seq_encoder_1": 17,
        }
        only_buy_feature_dims = {"only_buy_seq_encoder_0": 17}
        self.assertDictEqual(
            embedding_group.group_feature_dims("wide"), wide_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("deep"), deep_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("only_buy"), only_buy_feature_dims
        )

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(24))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(10)], dtype=torch.float32),
            lengths=torch.tensor([3, 3, 2, 2]),
        ).to_dict()

        if graph_type == TestGraphType.JIT_SCRIPT:
            embedding_group = create_test_module(
                _EGScriptWrapper(embedding_group), graph_type
            )
            result = embedding_group(
                sparse_features={BASE_DATA_GROUP: sparse_feature},
                dense_features={BASE_DATA_GROUP: dense_feature},
                sequence_dense_features=sequence_dense_feature,
            )
        else:
            embedding_group = create_test_module(embedding_group, graph_type)
            result = embedding_group(
                Batch(
                    sparse_features={BASE_DATA_GROUP: sparse_feature},
                    dense_features={BASE_DATA_GROUP: dense_feature},
                    sequence_dense_features=sequence_dense_feature,
                )
            )
        self.assertEqual(result["wide"].size(), (2, 8))
        self.assertEqual(result["deep"].size(), (2, 67))
        self.assertEqual(result["only_buy"].size(), (2, 17))
        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_sequence_embedding_group_impl_input_tile(self, graph_type) -> None:
        os.environ["INPUT_TILE"] = "2"
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)

        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(24))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([3, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            EMPTY_KJT,
            EMPTY_KT,
            2,
        )

        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_sequence_embedding_group_impl_input_tile_emb(self, graph_type) -> None:
        os.environ["INPUT_TILE"] = "3"
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)

        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor(list(range(4))),
            lengths=torch.tensor([1, 1, 1, 1]),
        )
        sparse_feature_user = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(10))),
            lengths=torch.tensor([3, 3, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([3, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            sparse_feature_user,
            EMPTY_KT,
            2,
        )

        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))


if __name__ == "__main__":
    unittest.main()