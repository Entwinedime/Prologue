"""Run native logical workers on one physical device and a host rendezvous."""

from sglang.srt.entrypoints.engine import Engine


class SingleDeviceEngine(Engine):
    @staticmethod
    def run_scheduler_process_func(server_args, port_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                                   pp_rank, dp_rank, pipe_writer):
        import torch
        import torch.distributed as distributed
        from sglang.srt.distributed import parallel_state
        from sglang.srt.managers.scheduler import run_scheduler_process

        device = torch.get_device_module(server_args.device)
        if device.device_count() != 1:
            raise ValueError("logical shard workers require exactly one visible accelerator")
        device.set_device(0)

        # SGLang reuses the initialized world when constructing its TP, PP and attention groups.
        world_size = server_args.tp_size * server_args.pp_size
        rank = pp_rank * server_args.tp_size + tp_rank
        distributed.init_process_group("gloo", init_method="env://", world_size=world_size, rank=rank)
        parallel_state.init_distributed_environment(world_size, rank, "env://", local_rank=0, backend="gloo")

        run_scheduler_process(server_args, port_args, 0, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                              pp_rank, dp_rank, pipe_writer)
