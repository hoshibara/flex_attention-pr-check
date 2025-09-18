import os
import re
import csv
import argparse
import fnmatch

all_deduped = set()
DEVICE = os.environ.get("DEVICE", "xpu").upper()


def process_onednn_log(file_path, output_dir):
    """
    Extracts lines containing "exec,gpu" from onednn.verbose.log.
    """
    output_filename = os.path.join(
        output_dir, os.path.basename(file_path) + ".filtered.log"
    )
    extracted_lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if "exec,gpu" in line:
                    extracted_lines.append(line)

        with open(output_filename, "w", encoding="utf-8") as outfile:
            for line in extracted_lines:
                outfile.write(line)
        print(f"Processed '{file_path}'. Filtered log saved to '{output_filename}'")
        return True
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")
        return False


def parse_time_string(time_str: str) -> float:
    time_str = time_str.strip().lower()
    if not time_str or time_str == "--":
        return 0.0

    match = re.match(r"([\d.]+)\s*(ns|us|ms|s)?", time_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "ns" or unit is None:
            return value
        if unit == "us":
            return value * 1000
        elif unit == "ms":
            return value * 1000 * 1000
        elif unit == "s":
            return value * 1000 * 1000 * 1000
    return 0.0


def process_torch_profile(file_path, output_dir):
    output_filename = os.path.join(
        output_dir, os.path.basename(file_path) + ".parsed.csv"
    )
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_line = None
    split_line = None
    split_line_idx = None

    for i, line in enumerate(lines):
        if "Name" in line and "Input Shapes" in line and "# of Calls" in line:
            header_line = line
            header_line_idx = i
            break

    if header_line is None:
        print(f"Header not found in {file_path}")
        return

    for j in range(header_line_idx + 1, len(lines)):
        if re.match(r"^----+\s+", lines[j]):
            split_line = lines[j]
            split_line_idx = j
            break

    if split_line_idx is None or split_line is None:
        print(f"Split line not found in {file_path}")
        return

    col_starts = [m.start() for m in re.finditer(r"\S+", split_line)]
    col_ends = col_starts[1:] + [len(split_line)]
    col_names = [
        header_line[start:end].strip() for start, end in zip(col_starts, col_ends)
    ]

    data_lines = []
    for line in lines[split_line_idx + 1 :]:
        if re.match(r"^----+\s+", line):
            break
        if not line.strip():
            continue
        data_lines.append(line.rstrip("\n"))

    results = []
    for line in data_lines:
        fields = []
        for i, (start, end) in enumerate(zip(col_starts, col_ends)):
            fields.append(line[start:end].strip())
        row = dict(zip(col_names, fields))

        row["Name"] = re.sub(r"<.*", "", row["Name"]).strip()

        results.append(
            {
                "Name": row.get("Name", ""),
                "Input Shapes": row.get("Input Shapes", "[]"),
                f"{DEVICE} total": parse_time_string(row.get(f"{DEVICE} total", "0.0")),
                "# of Calls": int(row.get("# of Calls", "0").replace(",", "")),
            }
        )

    global all_deduped
    # filter DEVICE total
    results = list(filter(lambda x: x[f"{DEVICE} total"] > 0, results))
    results = list(filter(lambda x: x["Input Shapes"] != "[]", results))
    deduped = sorted(
        set(
            (r["Name"], r["# of Calls"], r[f"{DEVICE} total"], r["Input Shapes"])
            for r in results
        )
    )
    all_deduped.update(deduped)

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Call Num", f"{DEVICE} total", "Input Shapes"])
        writer.writerows(deduped)

    print(f"Saved processed CSV to {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Process log files in a specified folder."
    )
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing the log files."
    )

    args = parser.parse_args()

    folder_path = args.folder_path

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found or is not a directory.")
        return

    output_dir = os.path.join(folder_path, "processed_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: '{output_dir}'")

    processed_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            if filename == "onednn.verbose.log":
                if process_onednn_log(file_path, output_dir):
                    processed_count += 1
            elif fnmatch.fnmatch(filename, "*.profile.log"):
                if process_torch_profile(file_path, output_dir):
                    processed_count += 1

    global all_deduped
    all_deduped_output_filename = os.path.join(output_dir, "all_deduped_results.csv")
    all_deduped = sorted(all_deduped)
    with open(all_deduped_output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Call Num", f"{DEVICE} total", "Input Shapes"])
        writer.writerows(all_deduped)

    if processed_count == 0:
        print(
            f"No target files ('onednn.verbose.log', 'token_x_profile.txt') found in '{folder_path}'."
        )
    else:
        print(f"Finished processing. Processed {processed_count} file(s).")


if __name__ == "__main__":
    main()
