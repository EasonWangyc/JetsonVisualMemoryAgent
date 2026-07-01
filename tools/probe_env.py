from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jetson_visual_memory.env_probe import format_env_json


def main() -> None:
    print(format_env_json())


if __name__ == "__main__":
    main()
