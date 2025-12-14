#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script duyệt từng dòng dataset và đánh giá
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Màu sắc cho terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_dataset(file_path):
    """Load dataset từ file JSONL"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples

def load_progress(progress_file):
    """Load tiến độ đã duyệt"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "current_index": 0,
        "reviews": {},
        "stats": {"approved": 0, "rejected": 0, "flagged": 0, "skipped": 0}
    }

def save_progress(progress_file, progress):
    """Lưu tiến độ"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def display_sample(sample, index, total, progress):
    """Hiển thị một sample"""
    clear_screen()

    # Header
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}  DUYỆT DATASET - Sample {index+1}/{total}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

    # Stats
    stats = progress.get("stats", {})
    print(f"  {Colors.GREEN}✓ Approved: {stats.get('approved', 0)}{Colors.ENDC}  "
          f"{Colors.RED}✗ Rejected: {stats.get('rejected', 0)}{Colors.ENDC}  "
          f"{Colors.YELLOW}⚑ Flagged: {stats.get('flagged', 0)}{Colors.ENDC}  "
          f"○ Skipped: {stats.get('skipped', 0)}")
    print(f"{Colors.HEADER}{'-'*80}{Colors.ENDC}\n")

    # Source info
    print(f"{Colors.CYAN}📄 Source:{Colors.ENDC} {sample.get('source_doc', 'N/A')}  "
          f"{Colors.CYAN}Chunk:{Colors.ENDC} {sample.get('chunk_id', 'N/A')}")

    # Eval info
    eval_score = sample.get('eval_score', 'N/A')
    eval_reason = sample.get('eval_reason', 'N/A')
    eval_method = sample.get('eval_method', 'N/A')

    score_color = Colors.GREEN if eval_score >= 8 else (Colors.YELLOW if eval_score >= 5 else Colors.RED)
    print(f"{Colors.CYAN}📊 Score:{Colors.ENDC} {score_color}{eval_score}/10{Colors.ENDC}  "
          f"{Colors.CYAN}Reason:{Colors.ENDC} {eval_reason}  "
          f"{Colors.CYAN}Method:{Colors.ENDC} {eval_method}")
    print()

    # Question
    print(f"{Colors.BOLD}{Colors.BLUE}❓ QUESTION:{Colors.ENDC}")
    print(f"   {sample.get('question', 'N/A')}")
    print()

    # Answer
    print(f"{Colors.BOLD}{Colors.GREEN}💬 ANSWER:{Colors.ENDC}")
    answer = sample.get('answer', 'N/A')
    # Wrap long answers
    words = answer.split()
    lines = []
    current_line = "   "
    for word in words:
        if len(current_line) + len(word) + 1 > 78:
            lines.append(current_line)
            current_line = "   " + word
        else:
            current_line += " " + word if current_line.strip() else "   " + word
    if current_line.strip():
        lines.append(current_line)
    print('\n'.join(lines))
    print()

    # Previous review if exists
    sample_key = str(index)
    if sample_key in progress.get("reviews", {}):
        review = progress["reviews"][sample_key]
        status = review.get("status", "")
        status_color = {
            "approved": Colors.GREEN,
            "rejected": Colors.RED,
            "flagged": Colors.YELLOW
        }.get(status, "")
        print(f"{Colors.CYAN}📝 Đã đánh giá:{Colors.ENDC} {status_color}{status.upper()}{Colors.ENDC}")
        if review.get("note"):
            print(f"   Note: {review['note']}")
        print()

    print(f"{Colors.HEADER}{'-'*80}{Colors.ENDC}")

def show_commands():
    """Hiển thị danh sách lệnh"""
    print(f"\n{Colors.BOLD}Lệnh:{Colors.ENDC}")
    print(f"  {Colors.GREEN}[Enter] hoặc [a]{Colors.ENDC} - Approve (chấp nhận)")
    print(f"  {Colors.RED}[r]{Colors.ENDC}               - Reject (từ chối)")
    print(f"  {Colors.YELLOW}[f]{Colors.ENDC}               - Flag (đánh dấu cần xem lại)")
    print(f"  [s]               - Skip (bỏ qua)")
    print(f"  [n]               - Next (sample tiếp theo)")
    print(f"  [p]               - Previous (sample trước)")
    print(f"  [g] <số>          - Go to (đi đến sample số X)")
    print(f"  [/] <từ khóa>     - Search (tìm kiếm)")
    print(f"  [e]               - Export (xuất kết quả)")
    print(f"  [q]               - Quit (thoát)")
    print()

def add_note():
    """Thêm ghi chú"""
    note = input(f"  {Colors.CYAN}Ghi chú (Enter để bỏ qua):{Colors.ENDC} ").strip()
    return note

def search_samples(samples, keyword):
    """Tìm kiếm samples theo keyword"""
    results = []
    keyword_lower = keyword.lower()
    for i, sample in enumerate(samples):
        if (keyword_lower in sample.get('question', '').lower() or
            keyword_lower in sample.get('answer', '').lower()):
            results.append(i)
    return results

def export_reviews(progress, samples, output_dir):
    """Xuất kết quả đánh giá"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Xuất approved
    approved_file = output_dir / f"approved_{timestamp}.jsonl"
    rejected_file = output_dir / f"rejected_{timestamp}.jsonl"
    flagged_file = output_dir / f"flagged_{timestamp}.jsonl"

    approved_count = 0
    rejected_count = 0
    flagged_count = 0

    with open(approved_file, 'w', encoding='utf-8') as fa, \
         open(rejected_file, 'w', encoding='utf-8') as fr, \
         open(flagged_file, 'w', encoding='utf-8') as ff:

        for idx_str, review in progress.get("reviews", {}).items():
            idx = int(idx_str)
            sample = samples[idx].copy()
            sample["review_status"] = review["status"]
            sample["review_note"] = review.get("note", "")

            if review["status"] == "approved":
                fa.write(json.dumps(sample, ensure_ascii=False) + "\n")
                approved_count += 1
            elif review["status"] == "rejected":
                fr.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rejected_count += 1
            elif review["status"] == "flagged":
                ff.write(json.dumps(sample, ensure_ascii=False) + "\n")
                flagged_count += 1

    print(f"\n{Colors.GREEN}Đã xuất:{Colors.ENDC}")
    print(f"  - {approved_file} ({approved_count} samples)")
    print(f"  - {rejected_file} ({rejected_count} samples)")
    print(f"  - {flagged_file} ({flagged_count} samples)")

def main():
    # Xác định file dataset
    if len(sys.argv) > 1:
        split_name = sys.argv[1]
    else:
        print(f"{Colors.BOLD}Chọn split để duyệt:{Colors.ENDC}")
        print("  1. train (3723 samples)")
        print("  2. validation (776 samples)")
        print("  3. test (786 samples)")
        choice = input("\nChọn (1/2/3): ").strip()
        split_map = {"1": "train", "2": "validation", "3": "test"}
        split_name = split_map.get(choice, "train")

    # Paths
    base_dir = Path(__file__).parent
    split_dir = base_dir / "output" / "split"
    dataset_file = split_dir / f"{split_name}.jsonl"
    progress_file = split_dir / f".review_progress_{split_name}.json"

    if not dataset_file.exists():
        print(f"{Colors.RED}Không tìm thấy file: {dataset_file}{Colors.ENDC}")
        return

    # Load data
    print(f"\n{Colors.CYAN}Đang load dataset...{Colors.ENDC}")
    samples = load_dataset(dataset_file)
    progress = load_progress(progress_file)
    total = len(samples)

    print(f"{Colors.GREEN}Loaded {total} samples từ {split_name}{Colors.ENDC}")
    print(f"Tiến độ hiện tại: đã duyệt {len(progress.get('reviews', {}))} samples")
    input("\nNhấn Enter để bắt đầu...")

    current_index = progress.get("current_index", 0)

    while True:
        if current_index < 0:
            current_index = 0
        if current_index >= total:
            current_index = total - 1

        sample = samples[current_index]
        display_sample(sample, current_index, total, progress)
        show_commands()

        try:
            cmd = input(f"{Colors.BOLD}>>> {Colors.ENDC}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            cmd = "q"

        if cmd == "" or cmd == "a":
            # Approve
            note = add_note()
            progress["reviews"][str(current_index)] = {
                "status": "approved",
                "note": note,
                "timestamp": datetime.now().isoformat()
            }
            progress["stats"]["approved"] = progress["stats"].get("approved", 0) + 1
            progress["current_index"] = current_index + 1
            save_progress(progress_file, progress)
            current_index += 1

        elif cmd == "r":
            # Reject
            note = add_note()
            progress["reviews"][str(current_index)] = {
                "status": "rejected",
                "note": note,
                "timestamp": datetime.now().isoformat()
            }
            progress["stats"]["rejected"] = progress["stats"].get("rejected", 0) + 1
            progress["current_index"] = current_index + 1
            save_progress(progress_file, progress)
            current_index += 1

        elif cmd == "f":
            # Flag
            note = add_note()
            progress["reviews"][str(current_index)] = {
                "status": "flagged",
                "note": note,
                "timestamp": datetime.now().isoformat()
            }
            progress["stats"]["flagged"] = progress["stats"].get("flagged", 0) + 1
            progress["current_index"] = current_index + 1
            save_progress(progress_file, progress)
            current_index += 1

        elif cmd == "s":
            # Skip
            progress["stats"]["skipped"] = progress["stats"].get("skipped", 0) + 1
            current_index += 1

        elif cmd == "n":
            # Next
            current_index += 1

        elif cmd == "p":
            # Previous
            current_index -= 1

        elif cmd.startswith("g ") or cmd.startswith("g"):
            # Go to
            try:
                if cmd.startswith("g "):
                    target = int(cmd[2:]) - 1
                else:
                    target = int(input("  Nhập số sample: ")) - 1
                if 0 <= target < total:
                    current_index = target
                else:
                    print(f"{Colors.RED}Số không hợp lệ (1-{total}){Colors.ENDC}")
                    input("Nhấn Enter...")
            except ValueError:
                print(f"{Colors.RED}Vui lòng nhập số{Colors.ENDC}")
                input("Nhấn Enter...")

        elif cmd.startswith("/"):
            # Search
            keyword = cmd[1:].strip()
            if not keyword:
                keyword = input("  Nhập từ khóa: ").strip()
            if keyword:
                results = search_samples(samples, keyword)
                if results:
                    print(f"\n{Colors.GREEN}Tìm thấy {len(results)} kết quả:{Colors.ENDC}")
                    for i, idx in enumerate(results[:20]):  # Hiển thị tối đa 20
                        q = samples[idx]['question'][:60] + "..." if len(samples[idx]['question']) > 60 else samples[idx]['question']
                        print(f"  {idx+1}. {q}")
                    if len(results) > 20:
                        print(f"  ... và {len(results) - 20} kết quả khác")
                    try:
                        go_to = input("\n  Đi đến kết quả số (Enter để bỏ qua): ").strip()
                        if go_to:
                            result_idx = int(go_to) - 1
                            if 0 <= result_idx < len(results):
                                current_index = results[result_idx]
                    except ValueError:
                        pass
                else:
                    print(f"{Colors.YELLOW}Không tìm thấy kết quả{Colors.ENDC}")
                    input("Nhấn Enter...")

        elif cmd == "e":
            # Export
            export_reviews(progress, samples, split_dir)
            input("\nNhấn Enter để tiếp tục...")

        elif cmd == "q":
            # Quit
            progress["current_index"] = current_index
            save_progress(progress_file, progress)
            print(f"\n{Colors.GREEN}Đã lưu tiến độ. Tạm biệt!{Colors.ENDC}")
            break

        else:
            print(f"{Colors.YELLOW}Lệnh không hợp lệ. Nhấn Enter để tiếp tục.{Colors.ENDC}")
            input()

    # Final stats
    print(f"\n{Colors.BOLD}Thống kê cuối:{Colors.ENDC}")
    stats = progress.get("stats", {})
    print(f"  ✓ Approved: {stats.get('approved', 0)}")
    print(f"  ✗ Rejected: {stats.get('rejected', 0)}")
    print(f"  ⚑ Flagged: {stats.get('flagged', 0)}")
    print(f"  ○ Skipped: {stats.get('skipped', 0)}")
    print(f"  Tổng đã duyệt: {len(progress.get('reviews', {}))}/{total}")

if __name__ == "__main__":
    main()
