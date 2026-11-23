"""Check the native reference boundary before comparing structure or performance."""


def validate_reference(manifest, publications, mode):
    expected_initial = {"kv": "empty", "workspace": "warmed"}
    if manifest["mode"] != mode or manifest["initial_state"] != expected_initial:
        description = "lightweight" if mode == "light" else "detailed"
        raise ValueError(f"comparison requires an independent warmed {description} reference")
    if manifest["clock"] != "monotonic_raw_ns":
        raise ValueError("native reference requires monotonic raw host timestamps")

    delivery = manifest["delivery"]
    origin = delivery["origin_raw_ns"]
    arrivals = {}
    for row in delivery["requests"]:
        rid = row["request"]
        timestamp = row["submitted_raw_ns"] - origin
        if rid in arrivals or timestamp < 0:
            raise ValueError(f"invalid native delivery for request {rid}")
        arrivals[rid] = timestamp

    expected = {row["id"]: row["output_tokens"] for row in manifest["workload"]["requests"]}
    if arrivals.keys() != expected.keys() or publications.keys() != expected.keys():
        raise ValueError("reference delivery and publication identities must match the workload")
    for rid, rows in publications.items():
        last_time, last_count = origin + arrivals[rid], 0
        for row in rows:
            if row["raw_ns"] < last_time or row["count"] <= last_count:
                raise ValueError(f"invalid native publication order for request {rid}")
            last_time, last_count = row["raw_ns"], row["count"]
        if last_count != expected[rid]:
            raise ValueError(f"incomplete native publication for request {rid}")
    return arrivals
