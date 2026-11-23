# Prologue

Prologue is a serving simulator for one model across multiple configurations and request workloads. The current entry point checks a task, orders its requests, and lists the configuration and workload pairs.

## Build and plan a task

Build the software-only toolchain and then install the C++23 core and Python CLI into the application image.

```bash
docker build -f docker/images/modeling/Dockerfile -t prologue-modeling:ubuntu24.04 .
docker build -f docker/images/application/Dockerfile -t prologue-app:ubuntu24.04 .
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$PWD/example/task.json",dst=/task.json,readonly \
  prologue-app:ubuntu24.04 plan /task.json
```

The [example task](example/task.json) declares one model, two configurations and two workloads. The command prints four targets and the ordered request inputs. It exits with a diagnostic when the task violates the input contract. Planning can run on a CPU host with just this input file. Native observation and prediction are subsequent stages.

## License

Prologue is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses.
