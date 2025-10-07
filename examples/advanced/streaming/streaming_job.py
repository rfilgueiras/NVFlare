# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

from src.streaming_controller import StreamingController
from src.streaming_executor import StreamingExecutor

from nvflare import FedJob
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from nvflare.app_common.streamers.file_retriever import FileRetriever
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer


def main():
    args = define_parser()
    retriever_mode = args.retriever_mode
    model_name = args.model_name

    # Create the FedJob
    job = FedJob(name="streaming", min_clients=1)

    if retriever_mode:
        if retriever_mode == "file":
            retriever = FileRetriever(source_dir="./", dest_dir="./")
            job.to_server(retriever, id="retriever")
            job.to_clients(retriever, id="retriever")
            job_dir = "/tmp/nvflare/workspace/jobs/file_streaming"
            work_dir = "/tmp/nvflare/workspace/works/file_streaming"
        elif retriever_mode == "container":
            retriever = ContainerRetriever()
            job.to_server(retriever, id="retriever")
            job.to_clients(retriever, id="retriever")
            job_dir = "/tmp/nvflare/workspace/jobs/container_streaming"
            work_dir = "/tmp/nvflare/workspace/works/container_streaming"
        elif retriever_mode == "tensor":
            server_streamer = TensorServerStreamer(format="numpy", tasks=["retrieve_model"], entry_timeout=60.0)
            job.to_server(server_streamer)
            client_streamer = TensorClientStreamer(format="numpy", tasks=["retrieve_model"], entry_timeout=60.0)
            job.to_clients(client_streamer)
            job_dir = "/tmp/nvflare/workspace/jobs/tensor_streaming"
            work_dir = "/tmp/nvflare/workspace/works/tensor_streaming"
        else:
            raise ValueError(f"invalid retriever_mode {retriever_mode}")

        controller = StreamingController(
            model_name=model_name, retriever_mode=retriever_mode, retriever_id="retriever", task_timeout=300
        )
        job.to_server(controller)
        executor = StreamingExecutor(retriever_mode=retriever_mode, retriever_id="retriever")
        job.to_clients(executor, tasks=["*"])
    else:
        job_dir = "/tmp/nvflare/workspace/jobs/regular_streaming"
        work_dir = "/tmp/nvflare/workspace/works/regular_streaming"
        controller = StreamingController(model_name=model_name)
        job.to_server(controller)
        executor = StreamingExecutor()
        job.to_clients(executor, tasks=["*"])

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", work_dir)
    job.simulator_run(work_dir, n_clients=1, threads=1)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever_mode",
        type=str,
        default=None,
        help="Retriever mode, default is None, can be 'container', 'file', or 'tensor'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/llama-3.2-1b",
        help="Model name, default is the large 'meta-llama/llama-3.2-1b', "
        "can be any pytorch model from HuggingFace, e.g. 'distilbert/distilgpt2' for a smaller model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
