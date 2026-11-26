"""Publish the evidence scope and independent action comparison results."""

import json

COMPARISON_SCOPE = (
    "Every invocation retains its run-local identity; cross-run occurrence pairs require either unique "
    "full logical work anchors or a checked consumed native prefix independently reconstructed from "
    "native CPU observations, protocol posts and input reads. Every actual consumed identity and source "
    "reference binds its instantiated graph; positions alone never establish correspondence. Without "
    "this witness, repeatable reads and unanchored control calls remain unpaired even if observed once. "
    "Logical matches retain ancestor contexts and exclude time-only observation placement. Structural "
    "classes omit ancestor state for common-content checks. Native ABIs whose every observed "
    "declaration is independently verified host-only also share CPU contents across queue snapshots and "
    "continuation cuts; their predicted body never selects a class, and logical/time identities remain "
    "unchanged. Classes never identify external producers or causal boundaries. Contents and "
    "within-action boundary reachability/work placement are checked separately for every predicted "
    "occurrence when their respective observed signatures are uniform. This does not pair external "
    "producers, effects, query outcomes or communication rounds. Every nonempty predicted external data "
    "input is separately checked against its own published storage/version export and the shared "
    "all-rank causal graph. Every explicit signal wait is checked against its actual generation and "
    "recorded direct signal edge; completion queries do not acquire a wait. These run-local checks do "
    "not require native occurrence pairs. Signal cross-run matched counts concern producer identity. "
    "All predicted communication edges are separately checked against run-local channel participants "
    "and causal order, including repeated calls; this does not establish cross-run occurrence pairing "
    "or payload correctness. A qualified full-context pair uses its own observed scope even when the "
    "structural class varies; unpaired variation remains unverified. No signature is selected by "
    "resemblance to the prediction. Cross-action order, data writers/export readiness and native "
    "protocol participant dependencies retain their independent checks. All causal comparisons use the "
    "same per-run all-rank graphs, including unnamed intermediates, dynamic prediction edges and "
    "verified model/CPU communication. A path may cross ranks and return. Missing protocol coverage "
    "withholds causal qualification rather than proving a negative from an isolated rank. Event "
    "numbers, raw edge counts and measured times are not matched. Differences require diagnosis, not "
    "automatic attribution to core bugs."
)


def write_action_report(output, report):
    output.mkdir(parents=True)
    (output / "actions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    checks, state, counts = report["action_checks"], report["state_lifetimes"], report["counts"]
    connections = report["connections"]
    reads = report["resource_effects"]["reads"]
    streams = report["stream_synchronization"]
    order = report["action_order"]
    wire = report["cpu_protocol"]["wire_payloads"]
    outside_flow = sum(
        value["input_reads"]["outside_flow"]
        for value in report["request_ingress"].values()
        if "input_reads" in value
    )
    rows = [
        "# Independent Execution Structure Comparison",
        "",
        f"Case outcome `{report['status']}`. Evidence coverage `{report['coverage']}`.",
        "",
        "The case outcome depends on the full evidence scope. Multiple trace differences can share a root cause.",
        "",
        "| Check | Coverage | Results and gaps |",
        "| --- | --- | --- |",
    ]

    for name, label in (
        ("contents", "Expanded template contents"),
        ("dependencies", "Named boundary dependencies within calls"),
    ):
        value = checks[name]
        rows.append(f"| {label} | {value['checked']} of {checks['predicted_occurrences']} predicted calls | "
                    f"{value['matched']} matched, {value['checked'] - value['matched']} differed, "
                    f"and {checks['predicted_occurrences'] - value['checked']} remain unchecked. "
                    "Shared content checks establish coverage at the template level. |")

    if wire["witnessed"]:
        rows.append(f"| CPU wire payloads | {wire['checked']} of {wire['witnessed']} witnessed messages | "
                    f"{wire['matched']} matched in dtype and byte geometry. |")

    if transfers := connections["device_transfer_payloads"]:
        rows.append(
            f"| Device transfer payloads | {connections['device_transfer_payloads_compared']} of {transfers} "
            "native endpoints | "
            f"{connections['device_transfer_payloads_matched']} matched in direction, dtype and shape. |"
        )

    rows.extend(
        [
            f"| Production and consumption | {connections['predicted_data_inputs_checked']} nonempty external "
            "data inputs and "
            f"{connections['predicted_signal_inputs_checked']} completion waits | "
            f"Readiness dependencies protect {connections['predicted_data_inputs_ready']} data inputs and "
            f"{connections['predicted_signal_inputs_ready']} waits. The check uses within-run evidence and "
            "treats queries as reads. |",
            f"| Dynamic stream waits | {streams.get('waits_checked', 0)} of {streams.get('declared_waits', 0)} "
            "waits | "
            f"{streams['matched_dependencies']} of {streams['required_dependencies']} queue frontier "
            "dependencies matched, "
            f"with {streams['extra_dependencies']} extra dependencies and {streams['activations_missing']} "
            "missing activations. "
            "Queue snapshots use the actual wait arrival. |",

            f"| Physical state within each run | {state.get('tensor_ports_checked', 0)} nonempty visible tensor "
            "ports, "
            f"{state.get('kv_regions_checked', 0)} logical KV regions, and "
            f"{state.get('field_bindings_checked', 0)} field bindings | "
            f"{state.get('kv_regions_unbound', 0)} KV regions and {state.get('field_bindings_missing', 0)} "
            "field accesses lack full bindings. "
            f"{state.get('pending_conflicts_ordered', 0)} of {state.get('pending_conflicts', 0)} in-flight "
            "conflicts have an ordering path. |",
            f"| Cross-run call pairing | {report['actions_compared']} paired calls | "
            f"{report['actions_contents_matched']} matched in contents and "
            f"{report['actions_dependencies_matched']} in internal dependencies. "
            f"{report['unpaired_calls']['native']} native and {report['unpaired_calls']['predicted']} predicted "
            "calls remain unpaired. |",

            f"| Execution trace | Minimal order relations and resource history from both runs | "
            f"{order.get('missing_relations', 0)} relations appear only in the native run and "
            f"{order.get('extra_relations', 0)} only in prediction. "
            f"Other differences include {order.get('alias_conflicts', 0)} event alias conflicts, "
            f"{counts.get('repeated_call_order', 0)} repeated call positions, and "
            f"{counts.get('resource_effects', 0)} resource effects. |",
            f"| Resource queries | {sum(value['count'] for value in reads['native'])} native and "
            f"{sum(value['count'] for value in reads['predicted'])} predicted scalar reads | "
            "The checks cover native getters and semantic boundaries. Per-call comparisons require paired "
            "evidence. |",
            "",

            "Control interactions require separate evidence for input delivery, queries, notifications, "
            "admission, and "
            "time conditions that affect decisions.",
            f"Input comparisons cover the serving interval and all nonempty request deliveries. "
            f"The trace retains {outside_flow} empty reads outside that interval.",
            "Unpaired calls, missing protocol evidence, and reuse paths that never triggered limit coverage. "
            "Verifying in-flight page reuse requires an observed in-flight return.",
            "`actions.json` contains all finding categories, representative issues, and the scope of each check.",
            "",
        ]
    )
    (output / "actions.md").write_text("\n".join(rows))
