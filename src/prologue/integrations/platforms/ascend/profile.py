"""Open a bounded Ascend task trace without a serving profiler lifecycle."""


def task_profile(directory):
    import torch_npu.profiler as sdk

    return sdk.profile(
        activities=[sdk.ProfilerActivity.CPU, sdk.ProfilerActivity.NPU],
        on_trace_ready=sdk.tensorboard_trace_handler(str(directory)),
        experimental_config=sdk._ExperimentalConfig(profiler_level=sdk.ProfilerLevel.Level1,
                                                    aic_metrics=sdk.AiCMetrics.AiCoreNone),
    )
