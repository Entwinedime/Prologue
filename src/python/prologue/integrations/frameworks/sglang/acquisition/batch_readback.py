"""Validate measured batch geometry, host effects and model operands for every consumer."""


class BatchReadback:
    rule = "sglang.ascend.scheduler_batch"

    @classmethod
    def observed_asset(cls, batch, boundary, facts):
        if boundary["phase"] != batch["phase"]:
            raise ValueError("native batch acquisition ran in a different phase")
        if "output" in boundary:
            cls.check_model(boundary, facts)
            return "structure/model.json"
        if boundary["phase"] == "admission":
            if boundary["waiting"] != [row["id"] for row in batch["requests"]]:
                raise ValueError("native waiting queue differs from the supplied request order")
        elif boundary["phase"] not in {"selection", "scheduler"}:
            cls.check_host(batch, boundary, facts)
        return "structure/host.json"

    @staticmethod
    def geometry(members):
        return [{k: member[k] for k in ("id", "phase", "context_tokens", "new_tokens")} for member in members]

    @staticmethod
    def check_model(bounds, facts):
        native = bounds["output"]
        count = len(bounds["members"])
        if len(native["tokens"]) != count or native["tokens"] != native["relayed"] or native["device"] != "npu":
            raise ValueError("each sampled device token must reach its original native request slot")
        scopes = facts["scopes"]
        root, = (i for i, s in enumerate(scopes) if s["role"] == "run_batch" and s["parent"] == 0)

        def tensor(role, direction, name):
            ports = [
                p["tensor"]
                for i, s in enumerate(scopes)
                if (i == root or s["parent"] == root) and s["role"] == role
                for p in s["ports"]
                if (p["direction"], p["name"]) == (direction, name)
            ]
            if len(ports) != 1:
                raise ValueError(f"one native model cut requires one {role} {direction} port for {name}")
            return ports[0]

        logits = tensor("model", "output", "next_token_logits")
        tokens = tensor("run_batch", "output", "next_token_ids")
        if (tensor("sample", "input", "next_token_logits")["value"] != logits["value"]
                or tensor("sample", "output", "next_token_ids")["value"] != tokens["value"]):
            raise ValueError("model, sampler and batch must hand off the observed tensor values")
        if tokens["shape"] != [count] or tokens["dtype"] != native["dtype"]:
            raise ValueError("the observed model output port must describe the returned token tensor")

    @staticmethod
    def check_host(batch, bounds, facts):
        expected = []
        for row, member in zip(batch["requests"], bounds["members"], strict=True):
            prompt = sum(n for _, n in row["prompt"])
            count = 1 if batch["phase"] == "prefill" else member["context_tokens"] - prompt + 2
            expected.append({"id": row["id"], "generated_tokens": count, "finished": count == row["output_tokens"]})
        publication = [
            effect | {"tokens": [row["next_token"]]} for effect, row in zip(expected, batch["requests"], strict=True)
        ]
        emitter = facts["environment"]["parallel"]["tp_rank"] == 0
        if bounds["effects"] != expected or bounds["publication"] != (publication if emitter else []):
            raise ValueError("native request state and receiving socket must agree on each produced token")
        scopes = facts["scopes"]
        bindings = [scope["bindings"] for scope in scopes]
        owners = {b.get("segment", {}).get("owner") for b in bindings}
        produced = [b["publication"] for b in bindings if "publication" in b]
        completed = [b["effects"] for b in bindings if "effects" in b and b["segment"]["owner"] == "result"]
        named = [{k: e[k] for k in ("id", "generated_tokens")} for e in expected]
        required = {"selection", "preparation", "result"} | ({"publication"} if emitter else set())
        if not required <= owners or produced != ([named] if emitter else []) or completed != [expected]:
            raise ValueError("observed host segments must account for the native request and publication effects")

        # The supplied token tensor is on the device. Its materialization must account for the actual batch payload.
        reads = [
            i
            for i, s in enumerate(scopes)
            if s["role"] == "host_read" and s["bindings"]["condition"]["input"] == "next_token_ids"
        ]
        copies = [
            s for s in facts["spans"] if s["scope"] in reads and s.get("copy", {}).get("operand") == "next_token_ids"
        ]
        if (
            len(reads) != 1
            or len(copies) != 1
            or copies[0]["copy"]["direction"] != "device_to_host"
            or copies[0]["operand_bytes"] != len(expected) * copies[0]["copy"]["element_bytes"]
            or not any(p["name"] == "next_token_ids" and p["direction"] == "input" for p in scopes[reads[0]]["ports"])
        ):
            raise ValueError("the host read must consume the supplied device output for each batch member")
