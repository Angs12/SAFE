# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.function_normalizer import FunctionNormalizer
from safetorch.parameters import Config


class SAFE(nn.Module):
    def __init__(self, config):
        super(SAFE, self).__init__()

        self.conf = config

        self.instructions_embeddings = torch.nn.Embedding(
            self.conf.num_embeddings, self.conf.embedding_size
        )

        self.bidirectional_rnn = torch.nn.GRU(
            input_size=self.conf.embedding_size,
            hidden_size=self.conf.rnn_state_size,
            num_layers=self.conf.rnn_depth,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        self.WS1 = Parameter(
            torch.Tensor(self.conf.attention_depth, 2 * self.conf.rnn_state_size)
        )
        self.WS2 = Parameter(
            torch.Tensor(self.conf.attention_hops, self.conf.attention_depth)
        )

        self.dense_1 = torch.nn.Linear(
            2 * self.conf.attention_hops * self.conf.rnn_state_size,
            self.conf.dense_layer_size,
            bias=True,
        )
        self.dense_2 = torch.nn.Linear(
            self.conf.dense_layer_size, self.conf.embedding_size, bias=True
        )

    def forward(self, instructions, lengths):
        if instructions.dim() == 1:
            instructions = instructions.unsqueeze(0)
            single = True
        else:
            single = False

        batch_size = instructions.shape[0]

        if lengths[0] <= 0:
            return torch.zeros(batch_size, self.conf.embedding_size)

        max_len = instructions.shape[1]
        instructions_vectors = self.instructions_embeddings(instructions)

        lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
        packed = pack_padded_sequence(
            instructions_vectors, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        output_packed, h_n = self.bidirectional_rnn(packed)
        output, _ = pad_packed_sequence(
            output_packed, batch_first=True, total_length=max_len
        )

        H = output

        ws1_tiled = self.WS1.unsqueeze(0).expand(batch_size, -1, -1)
        ws2_tiled = self.WS2.unsqueeze(0).expand(batch_size, -1, -1)

        A = torch.softmax(
            ws2_tiled.matmul(torch.tanh(ws1_tiled.matmul(H.transpose(1, 2)))), 2
        )

        M = A.matmul(H)

        flattened_M = M.view(
            batch_size, 2 * self.conf.attention_hops * self.conf.rnn_state_size
        )

        dense_1_out = F.relu(self.dense_1(flattened_M))
        function_embedding = F.normalize(self.dense_2(dense_1_out), dim=1, p=2)

        if single:
            function_embedding = function_embedding.squeeze(0)

        return function_embedding

    @classmethod
    def load(cls, model_dir, device="cpu", train=False):
        safe = cls(Config())
        safe.load_state_dict(
            torch.load(os.path.join(model_dir, "SAFEtorch.pt"), map_location=device)
        )
        safe = safe.to(device)
        normalizer = FunctionNormalizer(150)
        if train:
            for name, param in safe.named_parameters():
                if name.startswith("instructions_embeddings"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            return safe.train(), normalizer
        return safe.eval(), normalizer
