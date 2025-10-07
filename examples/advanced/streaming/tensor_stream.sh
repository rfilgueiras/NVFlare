pkill -9 python
bash utils/log_memory.sh >>/tmp/nvflare/workspace/tensor-stream.txt &
python streaming_job.py --retriever_mode tensor
