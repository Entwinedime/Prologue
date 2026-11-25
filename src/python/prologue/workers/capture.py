"""Run the capture, interpretation or export command supplied by the native image."""

from prologue.workers.registry import entry


def main():
    entry("observation.entry").main()


if __name__ == "__main__":
    main()
