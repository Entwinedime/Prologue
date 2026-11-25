"""Serve the control protocol through the framework declared by the native image."""

from prologue.workers.registry import entry


def main():
    entry("control.entry").main()


if __name__ == "__main__":
    main()
