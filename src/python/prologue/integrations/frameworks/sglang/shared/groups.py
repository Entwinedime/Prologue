"""Name actual native process groups consistently in observation and control."""

from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols


class ParallelGroups:
    def __init__(self):
        import torch.distributed as distributed
        get_parallel = resolve(symbols().get_parallel)

        parallel = get_parallel()
        self.groups = {"default": distributed.group.WORLD}
        for name in ("world", "tp", "pp", "attn_tp", "attn_cp", "moe_dp", "moe_ep", "moe_tp"):
            group = getattr(parallel, name + "_group")
            self.groups[name] = group.device_group
            self.groups[name + "_cpu"] = group.cpu_group

    def name(self, group):
        # Native aliases share the same process-group object, not merely the same participants.
        for name, candidate in self.groups.items():
            if candidate is group:
                return name
        raise ValueError("native communication uses a group outside the declared parallel topology")

    def is_cpu(self, group):
        # Device-free control also stores device metadata on CPU; native group roles retain the distinction.
        return any(candidate is group for name, candidate in self.groups.items() if name.endswith("_cpu"))

    def document(self, rank_offset):
        import torch.distributed as distributed

        return {
            name: {
                "role": self.name(group),
                "channel": f"collective/{group.group_name}",
                "members": [rank_offset + rank for rank in distributed.get_process_group_ranks(group)],
            }
            for name, group in self.groups.items()
        }


class WorkerRoles:
    role_fields = ("parallel", "kv_layout", "expert_layers", "kv_peers")

    @classmethod
    def bind_worker(cls, domain, role):
        return domain | {name: role[name] for name in cls.role_fields if name in role}

    @staticmethod
    def rank(domain):
        matches = [role["parallel"]["rank"] for role in domain["worker_roles"].values()
                   if all(role["parallel"].get(key) == value for key, value in domain["parallel"].items())]
        if len(matches) != 1:
            raise ValueError("work coordinates require one observed logical worker")
        return matches[0]
