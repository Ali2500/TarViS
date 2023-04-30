from typing import Dict, Any


def pretty_parse_dict(d: Dict[str, Any]) -> str:
    max_key_length = max([len(str(k)) for k in d.keys()])
    dash_line = f"{''.ljust(max_key_length, '-')}-----------------------------"

    print_lines = [
        dash_line,
        f"{'PARAM'.ljust(max_key_length)} | VALUE",
        dash_line
    ]

    for k, v in d.items():
        k = str(k).ljust(max_key_length, ' ')
        print_lines.append(f"{k} | {str(v)}")

    print_lines.append(dash_line)
    return "\n".join(print_lines)
