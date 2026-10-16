# Step-Level Sync for the Lightning Client API

Status: Proposed
Date: 2026-10-16

## Objective

Let one long `trainer.fit()` exchange the model with the server every N optimizer steps:

```python
import nvflare.client.lightning as flare

trainer = Trainer(strategy="ddp", devices=4, max_epochs=3, val_check_interval=10)
flare.patch(trainer, sync_every_n_steps=10)
trainer.fit(model, datamodule=dm)
```

At each sync, training pauses, sends the local model, loads the next global model in place and resumes with the
same optimizer, LR scheduler, AMP scaler and data position. Three hooks at the sync point let user code extend the
payload or react to a new global model without forking `FLCallback`.

## Background

`FLCallback` (`nvflare/app_opt/lightning/api.py`) receives in `on_train_start` and sends in `on_train_end`, so one FL
round is one `fit()` call. The shortest natural round is one epoch.

For large models, one epoch is far too long a round. One workload that motivates this design trains a large model with
DDP on multi-GPU nodes: one epoch takes about 5 hours, and the clients sync every 10 optimizer steps, about every
11 minutes, which is some 25 rounds per epoch.

The only way to get short rounds today is `Trainer(max_steps=N)` with one `fit()` per round. Each `fit()` then:

| Effect | Consequence |
| --- | --- |
| Starts a new epoch from the first batch | Without shuffling, the same first N batches train every round and the rest of the data is never seen; with shuffling, epochs stop meaning a pass over the data |
| Rebuilds the optimizer and LR scheduler | State survives only through `RestoreState`, which keeps a `deepcopy` of all optimizer state on the training device between rounds (2× the parameters for Adam) |
| Re-runs the per-fit setup | Dataloader workers, `setup`/`teardown`, and on GPU a round trip of the module to CPU (`pl_module.cpu()` before each send) |

Step-bounded rounds also hit a round-length bug, fixed separately (#5453).

## Goals

- `sync_every_n_steps` on `flare.patch()`; `None` keeps today's behaviour.
- Rounds of N optimizer steps inside one `fit()`, including across epoch boundaries and with gradient accumulation.
- No change to the optimizer, LR scheduler, AMP scaler, dataloader iterator or epoch counter at a sync.
- Validation configured on the trainer (`val_check_interval`, epoch end) keeps working and runs while the server
  aggregates.
- DDP: all ranks pause at the same step and hold identical weights after each sync.
- SCAFFOLD and FedProx work per sync as they work per round today.
- Three hooks at the sync point, in both modes.

## Non-Goals

- `train_with_evaluation` in step mode (see [`train_with_evaluation`](#train_with_evaluation)).
- Step sync across several `fit()` calls.
- Sharded strategies (FSDP, DeepSpeed); they are not supported by the patched path today either.
- Workload-specific behaviour: receive heartbeats, restart sentinels, payload dtype normalisation. The hooks let user
  code add these.

## Proposed API

```python
def patch(
    trainer: pl.Trainer,
    restore_state: bool = True,
    load_state_dict_strict: bool = True,
    update_fit_loop: bool = True,
    sync_every_n_steps: Optional[int] = None,
): ...
```

`sync_every_n_steps`:

- `None` (default): one round per `fit()`, as today.
- A positive `int`: a round ends every N optimizer steps (`trainer.global_step`), so gradient accumulation is handled
  and the count matches `NUM_STEPS_CURRENT_ROUND`. Anything else raises `ValueError` (`bool` included).

In step mode:

- The client script calls `fit()` once.
- `update_fit_loop` has no effect: the trainer's `max_epochs`/`max_steps` bound the whole training, not a round.
- `restore_state` has no effect and `RestoreState` is not installed: the optimizer lives through the single `fit()`.
- `train_with_evaluation=True` in the job config raises `ValueError` when `patch()` runs.
- The server's `num_rounds` should be `ceil(total optimizer steps / N)`. If the server finishes first, training stops
  at the next sync; if the trainer finishes first, the last round is shorter than N steps.

## Sync Lifecycle

```text
fit() ─ on_train_batch_start (first batch): receive round 1, load, start round
        │
        ├─ batch … ─ on_train_batch_end: N steps since round start → finish round, rank 0 sends
        │      (validation from val_check_interval / epoch end runs here, while the server aggregates)
        ├─ on_train_batch_start: rank 0 receives → broadcast → every rank loads in place → start round
        │      (no model → trainer.should_stop = True; a non-train task → stop, leave it pending)
        ├─ …
        └─ on_train_end: send the open round, if any, after the final validation
```

Rules:

1. **Sync condition.** At `on_train_batch_end`, the round finishes when
   `trainer.global_step - round_start_step >= sync_every_n_steps`. With gradient accumulation this first holds on the
   batch that ran the optimizer step.
2. **Send before validation.** The send happens at batch end and the receive at the next batch start. Lightning runs
   in-fit validation in between, so it overlaps the server's aggregation instead of delaying the upload.
3. **Last batch.** If the batch is the last of training (`global_step` reached `max_steps`, or last batch of the last
   epoch), the sync is skipped and `on_train_end` sends the round after the final validation and checkpointing. The
   client is then not stopped mid-validation when the server finishes. When training stops for another reason
   (`EarlyStopping`, `max_time`), the round is sent at its last sync or at `on_train_end`, whichever comes first.
4. **No send without a receive.** NVFlare rejects a send that no receive preceded. A round is open only between a
   receive that returned a train task and the next send; `on_train_end` sends only an open round.
5. **Receive results.**
   - A train task: load in place, start the next round.
   - No model (the server finished, or the job stopped): set `trainer.should_stop` and raise `StopIteration`, so
     the batch that has started does not train; no round is open. `should_stop` alone would still train that
     batch, one step that is never sent. `TrainingEpochLoop.run` ends the epoch on `StopIteration`, which keeps
     the LR scheduler in step with the optimizer. Returning `-1` from `LightningModule.on_train_batch_start` is
     the documented way to skip a batch, but a callback can only do it by wrapping the user's module, and it
     steps step-interval LR schedulers once more. `StopIteration` is not a documented contract, so a test pins
     the step count and the scheduler position.
   - A non-train task (evaluate, submit model): stop the same way, without loading it or clearing the Client API
     cache. The task stays pending, and the script can handle it after `fit()` with the usual loop:

     ```python
     trainer.fit(model, datamodule=dm)
     while flare.is_running():
         flare.receive()
         trainer.validate(model, datamodule=dm)
     ```
6. **In-place load.** The received weights go through the existing `_update_model` (key and shape checks, strict and
   non-strict loading) into the existing parameters with `load_state_dict`. No optimizer or loop state is touched.
7. **DDP.** All ranks evaluate the sync condition on the same `global_step`, so they reach the send and the receive
   at the same batch. Rank 0 sends and receives; the received model is broadcast with `trainer.strategy.broadcast`,
   as `FLCallback._receive_model` does today.
8. **Payload.** Rank 0 builds `FLModel(params={k: v.detach().cpu() ...})` from `pl_module.state_dict()`. The module
   stays on its device. Fit-per-round mode keeps `pl_module.cpu()`, so existing users see no change in where the
   module lives after `fit()`.
9. **Algorithms.** Each sync runs the existing `_AlgorithmHandlerManager.finish_round` (SCAFFOLD control diff, step
   count) before the send, and each receive runs `start_round`.
10. **Metadata.** `__fl_meta__` is read at every send, as at the end of `fit()` today, with the same conflict check
    against algorithm metadata.

## Hooks

Custom training code often has to act at the sync point:

| Need | When |
| --- | --- |
| Add EMA weights or other state to the outgoing payload | before send |
| Log drift between local and global weights | after receive, before load |
| Take entries that are not module weights (EMA) out of the payload and load them | after receive, before load |
| Refresh the EMA from the new global weights | after load |

Three hooks, defined as methods on any `pl.Callback` in `trainer.callbacks`, called when present:

| Hook | Arguments | Called | May |
| --- | --- | --- | --- |
| `on_fl_before_send(trainer, pl_module, fl_model)` | `fl_model` is `None` on ranks other than 0 | before every send | edit `fl_model.params` / `meta` |
| `on_fl_receive(trainer, pl_module, fl_model)` | the broadcast model | after receive, before load, once per received model | edit `fl_model` before NVFlare loads it |
| `on_fl_model_loaded(trainer, pl_module, fl_model)` | the loaded model | after load | read the loaded module |

- Called on every rank in the same order, so a hook may use collectives.
- Called in both modes. In fit-per-round mode, `on_fl_before_send` also runs before the metrics-only send of an
  evaluate task, where `fl_model.params` is `None`.
- Not called for an empty receive or for a non-train task left pending in step mode.
- Called in `trainer.callbacks` order.

```python
class EMASync(pl.Callback):
    def on_fl_before_send(self, trainer, pl_module, fl_model):
        if fl_model is not None:
            fl_model.meta["ema"] = {k: v.cpu() for k, v in pl_module.ema.state_dict().items()}

    def on_fl_receive(self, trainer, pl_module, fl_model):
        self._global_ema = fl_model.meta.pop("ema", None)

    def on_fl_model_loaded(self, trainer, pl_module, fl_model):
        if self._global_ema is not None:
            pl_module.ema.load_state_dict(self._global_ema)


trainer = Trainer(strategy="ddp", devices=4, max_epochs=3, val_check_interval=10, callbacks=[EMASync()])
flare.patch(trainer, sync_every_n_steps=10)
trainer.fit(model, datamodule=dm)
```

The hooks expose the same points that `_AlgorithmHandlerManager` already uses internally for SCAFFOLD and FedProx.

### Alternatives considered

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Methods on trainer callbacks (chosen)** | The Lightning way to add behaviour; several callbacks compose; no new `patch()` arguments; same duck typing as `__fl_meta__` | A misspelled hook name is silently ignored |
| B. Callables passed to `patch(...)` | Explicit in the signature | One function per hook; several consumers must be combined by hand |
| C. Subclass `FLCallback` | No new mechanism | `patch()` builds the callback, so it would need a way to inject the subclass; users depend on internals |
| Two hooks (merge `on_fl_receive` into `on_fl_model_loaded`) | Smaller surface | No way to log drift or take entries out of the payload before the load |

## `train_with_evaluation`

With `train_with_evaluation`, the server expects each update to carry the metrics of the received global model on
local data and uses them for model selection. In fit-per-round mode they come from `trainer.validate()` before
`fit()`. Step mode has no such call: in-fit validation evaluates the local model, and `FLCallback` already keeps in-fit
metrics out of these metrics. Validating the global model after each receive would mean running Lightning's
validation loop from inside `fit()`, which `trainer.validate()` does not support, and would delay training by one
validation pass per sync. Step mode therefore raises `ValueError` for `train_with_evaluation=True`; a global-model
validation per sync can follow if there is demand.

## Code Structure

The change stays in `FLCallback`; there is no subclass. The hooks run in both modes, step mode reuses the receive
checks, the algorithm handlers, the metadata handling and the step counting, and code that finds the callback with
`isinstance(cb, FLCallback)` keeps working.

1. Extract `_start_round` from `on_train_start` and `_finish_round_and_send` from `on_train_end`, with no behaviour
   change.
2. Step mode: `on_train_start` does nothing; `on_train_batch_start` receives, loads and calls `_start_round` when no
   round is open, including the first one; `on_train_batch_end` checks the sync condition and calls
   `_finish_round_and_send`; `on_train_end` sends an open round. `reset_state` does not change
   `max_epochs`/`max_steps` in step mode.
3. A helper calls the hooks on `trainer.callbacks`.
4. `patch()` validates `sync_every_n_steps`, passes it to `FLCallback`, and skips `RestoreState` in step mode.

## Memory

Per sync, with M the size of the state dict:

| Phase | Step mode | Fit-per-round mode, today |
| --- | --- | --- |
| Send | Rank 0: a CPU copy, +M host memory until the send returns; GPU unchanged; other ranks: nothing | `pl_module.cpu()` on every rank moves the module to host memory; the next `fit()` moves it back |
| Receive (rank 0) | The received model, +M host memory, held in the Client API cache until the round is sent | Same |
| Broadcast | `broadcast_object_list` pickles the model: about +M host memory on each rank; with NCCL, the pickled bytes are staged on each rank's GPU, about +M GPU memory for the duration of the broadcast | Same, once per round at `fit()` start |
| Load | `load_state_dict` copies into the existing parameters; no new GPU memory | Same |
| Optimizer | Lives through the single `fit()`; no copy | `RestoreState` keeps a `deepcopy` of the optimizer state on the training device between rounds |
| FedProx | Keeps the global weights on the training device during each round (+M GPU), as today | Same |

Payload extras added through `on_fl_before_send` (EMA weights) add their own size to the send.

The main risk is the temporary +M GPU memory of the broadcast: in step mode it happens every N steps, at a batch
boundary, with the optimizer state and the allocator's cached activation blocks resident. A callback with the same
broadcast every 10 steps runs in production on the workload above, so the existing broadcast is kept.

### Alternative: per-tensor broadcast

Rank 0 loads the received weights, then every parameter and buffer is broadcast in place with
`torch.distributed.broadcast`, as DDP syncs a module at start-up. This needs no pickling and no extra GPU memory, but:

1. It covers the module weights only. `meta`, `current_round` and the task flags still need the object broadcast, and
   for SCAFFOLD `meta` holds the global controls, about M again.
2. Ranks other than 0 would not see the payload, which breaks `on_fl_receive` on every rank.
3. The key and shape checks would run on rank 0 only. An error there leaves the other ranks blocked in the broadcast
   until the NCCL timeout, unless a status is broadcast first.
4. With `load_state_dict_strict=False`, the other ranks need the list of loaded keys.
5. It is only correct for replicated weights; `trainer.strategy.broadcast` works for any strategy.
6. One collective per tensor is slow for models with thousands of tensors; bucketing needs a private PyTorch helper
   (`_broadcast_coalesced`) or custom code.

It stays the documented fallback if the broadcast's GPU memory causes out-of-memory errors.

## Example

`examples/advanced/multi-gpu/lightning/client.py` gets a `--sync_every_n_steps` flag. Without it, the script keeps its
round loop; with it:

```python
trainer = Trainer(max_epochs=3, strategy="ddp", devices=2, val_check_interval=50)
flare.patch(trainer, sync_every_n_steps=50)
trainer.fit(model, datamodule=dm)
```

The Lightning section of `docs/user_guide/data_scientist_guide/client_api_usage.rst` documents the parameter, the
`num_rounds` rule and the hooks.

## Testing Plan

Unit tests in `tests/unit_test/app_opt/lightning/api_test.py`, with a real `pl.Trainer` on CPU and the Client API
mocked at the `receive`/`send` seam, as the existing `test_real_lightning_*` tests do. The `receive`/`send` mock
enforces NVFlare's exchange order: a send without a preceding receive fails.

| Expected outcome | Test |
| --- | --- |
| One `fit()` sends every N optimizer steps with `NUM_STEPS_CURRENT_ROUND == N`, including with gradient accumulation | count sends and step metadata |
| Each received model is what trains next | distinct weights per round, checked at the next batch |
| Optimizer, LR scheduler and data order carry over across syncs | same optimizer object and momentum buffers, `lr_scheduler.last_epoch`, uninterrupted sample sequence |
| No send without a receive at epoch boundaries or at the end of training | epoch ending on a sync, epoch ending mid-round, epochs shorter than N, a sync on the last batch |
| In-fit validation runs between send and receive; the last round is sent after the last validation | call order of send, validation and receive |
| The server finishing first stops training cleanly | `receive()` returns `None` |
| A non-train task mid-fit stops training and is handled by a later `trainer.validate()` | evaluate task after a sync |
| SCAFFOLD and FedProx produce per-sync metadata | existing algorithm fixtures in step mode |
| Hooks are called in order, once per received model, in both modes; edits take effect | recording callback; a hook that adds to `meta` and one that edits the received params |
| Invalid `sync_every_n_steps` and `train_with_evaluation` in step mode are rejected | `ValueError` from `patch()` |
| Fit-per-round mode is unchanged | the existing tests |
| Under DDP, all ranks pause at the same step, hold identical weights after each sync, only rank 0 sends, no hang | 2-rank CPU `ddp_spawn` (gloo) test |

End to end: `examples/advanced/multi-gpu/lightning` in the simulator with `--sync_every_n_steps`, recording peak GPU
memory per sync.
