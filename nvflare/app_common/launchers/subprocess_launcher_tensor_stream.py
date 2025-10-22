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
import shlex
import subprocess
from threading import Thread

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path

from .subprocess_launcher import SubprocessLauncher, log_subprocess_output


class SubprocessLauncherTensorStream(SubprocessLauncher):
    """A launcher that launches jobs as subprocesses with Tensor Stream support."""

    def _start_external_process(self, fl_ctx: FLContext):
        with self._lock:
            if self._process is None:
                command = self._script
                env = os.environ.copy()
                # TODO: add this as a parameter to the launcher
                # it should avoid duplicated code to use ExProcessTensorStreamClientAPI
                env["CLIENT_API_TYPE"] = "EX_PROCESS_API_TENSOR_STREAM"

                workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
                app_custom_folder = workspace.get_app_custom_dir(job_id)
                add_custom_dir_to_path(app_custom_folder, env)

                command_seq = shlex.split(command)
                self._process = subprocess.Popen(
                    command_seq, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self._app_dir, env=env
                )
                self._log_thread = Thread(target=log_subprocess_output, args=(self._process, self.logger))
                self._log_thread.start()
