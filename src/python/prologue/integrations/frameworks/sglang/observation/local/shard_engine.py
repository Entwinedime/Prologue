"""Run native logical workers on one physical device and a host rendezvous."""

from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols

Engine = resolve(symbols().engine)



class SingleDeviceEngine(Engine):
    @staticmethod
    def run_scheduler_process_func(server_args, port_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                                   pp_rank, dp_rank, pipe_writer):
        import torch
        run_scheduler_process = resolve(symbols().run_scheduler_process)

        from prologue.integrations.frameworks.sglang.observation.local.shard_transport import ShardTransport

        device = torch.get_device_module(server_args.device)
        if device.device_count() != 1:
            raise ValueError("logical shard workers require exactly one visible accelerator")
        device.set_device(0)

        world_size = server_args.tp_size * server_args.pp_size
        rank = pp_rank * server_args.tp_size + tp_rank
        ShardTransport.world(world_size, rank, "env://")

        run_scheduler_process(server_args, port_args, 0, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                              pp_rank, dp_rank, pipe_writer)
