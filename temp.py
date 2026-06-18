from pathlib import Path

import torch


CHECKPOINT_DIR = Path("models/specialized")
REPORT_PATH = Path("specialized_checkpoint_report.txt")


def format_value(value, indent=""):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"mean={value.float().mean().item():.6f}, std={value.float().std().item():.6f})"
        )

    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            lines.append(f"{indent}{key}: {format_value(item, indent + '  ')}")
        return "\n".join(lines)

    return repr(value)


def summarize_state_dict(state_dict):
    lines = []
    total_parameters = 0
    for name, tensor in state_dict.items():
        total_parameters += tensor.numel()
        lines.append(
            f"    {name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, numel={tensor.numel()}"
        )
    return total_parameters, lines


def write_specialized_checkpoint_report(
    checkpoint_dir=CHECKPOINT_DIR,
    report_path=REPORT_PATH,
):
    checkpoint_paths = sorted(Path(checkpoint_dir).glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    report_lines = [
        f"Specialized checkpoint report",
        f"checkpoint_dir: {Path(checkpoint_dir).resolve()}",
        f"checkpoint_count: {len(checkpoint_paths)}",
        "",
    ]

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        report_lines.append("=" * 100)
        report_lines.append(f"file: {checkpoint_path.name}")
        report_lines.append(f"size_bytes: {checkpoint_path.stat().st_size}")

        state_dict = checkpoint.get("state_dict", {})
        total_parameters, state_lines = summarize_state_dict(state_dict)

        for key, value in checkpoint.items():
            if key == "state_dict":
                report_lines.append(
                    f"state_dict: {len(state_dict)} tensors, {total_parameters} parameters"
                )
            else:
                report_lines.append(f"{key}: {format_value(value)}")

        report_lines.append("state_dict_tensors:")
        report_lines.extend(state_lines)
        report_lines.append("")

    Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    write_specialized_checkpoint_report()
