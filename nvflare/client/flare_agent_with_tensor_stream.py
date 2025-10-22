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

import time
import traceback
from typing import Any, Optional

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_constant import ReturnCode as RC
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler

from .flare_agent import AgentClosed, CallStateError, Task, _TaskContext
from .flare_agent_with_fl_model import FlareAgentWithFLModel


class TensorStreamFlareAgent(FlareAgentWithFLModel):

    def __init__(
        self,
        pipe: Optional[Pipe] = None,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=60.0,
        resend_interval=2.0,
        max_resends=None,
        submit_result_timeout=60.0,
        metric_pipe: Optional[Pipe] = None,
        tensors_pipe: Optional[Pipe] = None,
        task_channel_name: str = PipeChannelName.TASK,
        metric_channel_name: str = PipeChannelName.METRIC,
        tensors_channel_name: str = PipeChannelName.TENSOR,
        close_pipe: bool = True,
        close_metric_pipe: bool = True,
        decomposer_module: str = None,
    ):
        """FLARE agent that supports tensor stream via a dedicated pipe."""
        super().__init__(
            pipe=pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            submit_result_timeout=submit_result_timeout,
            metric_pipe=metric_pipe,
            task_channel_name=task_channel_name,
            metric_channel_name=metric_channel_name,
            close_pipe=close_pipe,
            close_metric_pipe=close_metric_pipe,
            decomposer_module=decomposer_module,
        )

        self.tensors_pipe = tensors_pipe
        self.tensors_pipe_handler = None
        if self.tensors_pipe:
            self.tensors_pipe_handler = PipeHandler(
                pipe=self.tensors_pipe,
                read_interval=read_interval,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                resend_interval=resend_interval,
                max_resends=max_resends,
            )

    def get_task(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Get a task from FLARE. This is a blocking call.

        Args:
            timeout (float, optional): If specified, this call is blocked only for the specified amount of time.
                If not specified, this call is blocked forever until a task has been received or agent has been closed.

        Returns:
            None if no task is available before timeout; or a Task object if task is available.

        Raises:
            AgentClosed exception if the agent has been closed before timeout.
            CallStateError exception if the call has not been made properly.
            AgentAbortException: If the other endpoint of the pipe requests to abort.
            AgentEndException: If the other endpoint has ended.
            AgentPeerGoneException: If the other endpoint is gone.

        Note: the application must make the call only when it is just started or after a previous task's result
        has been submitted.

        """
        if not self.pipe_handler:
            raise RuntimeError("task pipe is not available")
        start_time = time.time()
        while True:
            if self.asked_to_stop:
                raise AgentClosed("agent closed")

            if self.current_task:
                raise CallStateError("application called get_task while the current task is not processed")

            if timeout is not None and time.time() - start_time >= timeout:
                self.logger.debug("get request timeout")
                return None

            req: Optional[Message] = self.pipe_handler.get_next()
            if req is not None:
                if not isinstance(req.data, Shareable):
                    self.logger.error(f"bad task: expect request data to be Shareable but got {type(req.data)}")
                    raise RuntimeError("bad request data")

                # get task data as tensors
                tensors = self.get_tensors(task_id)
                # merge tensors into task.data
                task_data = tensors

                shareable = req.data
                task_data = self.shareable_to_task_data(shareable)
                task_id = shareable.get_header(FLContextKey.TASK_ID)
                task_name = shareable.get_header(FLContextKey.TASK_NAME)

                tc = _TaskContext(
                    task_id=task_id,
                    task_name=task_name,
                    msg_id=req.msg_id,
                )
                self.current_task = tc
                return Task(task_name=tc.task_name, task_id=tc.task_id, data=task_data)
            time.sleep(0.5)

    def get_tensors(self, task_id: str) -> Any:
        """Get tensors for the given task_id from FLARE.

        Args:
            task_id (str): The ID of the task to get tensors for.
        Returns:
            The tensors associated with the task_id.
        """
        # implement tensor retrieval via tensor stream pipe
        # using the tensor stream pipe read operations
        # until all tensors for the task_id are received
        pass

    def submit_result(self, result, rc=RC.OK) -> bool:
        """Submit the result of the current task.

        This is a blocking call. The agent will try to send the result to flare site until it is successfully sent or
        the task is aborted or the agent is closed.

        Args:
            result: result to be submitted
            rc: return code

        Returns:
            whether the result is submitted successfully

        Raises:
            the CallStateError exception if the submit_result call is not made properly.

        Notes: the application must only make this call after the received task is processed. The call can only be
        made a single time regardless whether the submission is successful.

        """
        if not self.pipe_handler:
            raise RuntimeError("task pipe is not available")
        with self.task_lock:
            current_task = self.current_task
            if not current_task:
                self.logger.error("submit_result is called but there is no current task!")
                return False

        self.submit_tensors(result, current_task.task_id)

        try:
            result = self._do_submit_result(current_task, result, rc)
        except Exception as ex:
            self.logger.error(f"exception submitting result to {current_task.sender}: {ex}")
            traceback.print_exc()
            result = False

        with self.task_lock:
            self.current_task = None

        return result

    def submit_tensors(self, result: Any, task_id: str) -> bool:
        """Submit tensors for the given task_id to FLARE.

        Args:
            result (Any): The result to submit.
            task_id (str): The ID of the task to submit tensors for.
        Returns:
            whether the result is submitted successfully.
        """
        # results are FLModel with tensors in params
        # we should use a generator to chunk the tensors and send via pipe
        # using the tensor stream pipe write operations
        pass
