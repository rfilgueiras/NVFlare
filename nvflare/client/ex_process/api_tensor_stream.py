# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Optional

from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.client.flare_agent_with_tensor_stream import TensorStreamFlareAgent
from nvflare.client.model_registry import ModelRegistry

from .api import ExProcessClientAPI, _create_client_config, _create_pipe_using_config


class ExProcessTensorStreamClientAPI(ExProcessClientAPI):

    def __init__(self, config_file: str):
        super().__init__(config_file=config_file)

    def init(self, rank: Optional[str] = None):
        """Initializes NVFlare Client API environment.

        Args:
            rank (str): local rank of the process.
                It is only useful when the training script has multiple worker processes. (for example multi GPU)
        """

        if rank is None:
            rank = os.environ.get("RANK", "0")

        if self.model_registry:
            self.logger.warning("Warning: called init() more than once. The subsequence calls are ignored")
            return

        client_config = _create_client_config(config=self.config_file)

        flare_agent = None
        try:
            if rank == "0":
                if client_config.get_exchange_format() == ExchangeFormat.PYTORCH:
                    pass
                    # _register_tensor_decomposer()

                pipe, task_channel_name = None, ""
                if ConfigKey.TASK_EXCHANGE in client_config.config:
                    pipe, task_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.TASK_EXCHANGE
                    )
                metric_pipe, metric_channel_name = None, ""
                if ConfigKey.METRICS_EXCHANGE in client_config.config:
                    metric_pipe, metric_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.METRICS_EXCHANGE
                    )

                tensors_pipe, tensors_channel_name = None, ""
                if ConfigKey.TENSORS_EXCHANGE in client_config.config:
                    tensors_pipe, tensors_channel_name = _create_pipe_using_config(
                        client_config=client_config, section=ConfigKey.TENSORS_EXCHANGE
                    )

                flare_agent = TensorStreamFlareAgent(
                    pipe=pipe,
                    task_channel_name=task_channel_name,
                    metric_pipe=metric_pipe,
                    metric_channel_name=metric_channel_name,
                    tensors_pipe=tensors_pipe,
                    tensors_channel_name=tensors_channel_name,
                    heartbeat_timeout=client_config.get_heartbeat_timeout(),
                )
                flare_agent.start()

            self.model_registry = ModelRegistry(client_config, rank, flare_agent)
            self.flare_agent = flare_agent
        except Exception as e:
            self.logger.error(f"flare.init failed: {e}")
            raise e
