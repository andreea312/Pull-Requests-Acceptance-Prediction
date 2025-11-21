import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from github_utils import fetch_single_pr, _safe_github_call
from feature_extraction import build_initial_pr_dataframe
from config import REPOS, OUTPUT_DIR

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

COLUMNS = [
    "number", "title", "created_at", "closed_at", "merged_at", "additions", "deletions",
    "changed_files", "commits", "author", "comments_list", "reviewers_list", "commits_list",
    "files_metrics", "min_max_nesting", "avg_max_nesting", "max_max_nesting", "min_func_count",
    "avg_func_count", "max_func_count", "min_max_args", "avg_max_args", "max_max_args",
    "min_call_count", "avg_call_count", "max_call_count", "min_if_count", "avg_if_count",
    "max_if_count", "min_loop_count", "avg_loop_count", "max_loop_count", "min_avg_cc",
    "avg_avg_cc", "max_avg_cc", "min_max_cc", "avg_max_cc", "max_max_cc", "min_loc", "avg_loc",
    "max_loc", "min_lloc", "avg_lloc", "max_lloc", "min_sloc", "avg_sloc", "max_sloc",
    "min_comments", "avg_comments", "max_comments", "min_multi_comments", "avg_multi_comments",
    "max_multi_comments", "min_blank", "avg_blank", "max_blank", "title_length",
    "description_length", "files_with_content", "is_bugfix", "is_refactor", "is_feature"
]

TARGET_PRS_WITH_PYTHON = 2500
FETCH_BATCH_SIZE = 200
SAVE_EVERY = 25
MAX_WORKERS = 10

csv_lock = Lock()


def fetch_pr_list_descending(repo_full_name, max_pr_number=None, max_prs=1000):
    repo = _safe_github_call(lambda c, name: c.get_repo(name), repo_full_name)
    pulls = repo.get_pulls(state="closed", sort="created", direction="desc")

    all_prs = []
    page = 0

    pbar = tqdm(total=max_prs, desc=f"Fetching PRs below #{max_pr_number if max_pr_number else 'latest'}", unit="PR")

    while len(all_prs) < max_prs:
        page_items = _safe_github_call(lambda c, pulls, p: pulls.get_page(p), pulls, page)
        if not page_items:
            break

        for pr in page_items:
            # Skip already processed PRs
            if max_pr_number is not None and pr.number >= max_pr_number:
                continue

            # Skip merged PRs
            if pr.merged_at is not None:
                continue

            all_prs.append(pr)
            pbar.update(1)

            if len(all_prs) >= max_prs:
                break

        page += 1

    pbar.close()
    print(f"[INFO] Retrieved {len(all_prs)} closed-unmerged PRs")
    return all_prs


def process_repository(repo_name: str):
    print(f"\n Processing repository: {repo_name} ===")

    # Create repository-specific directory
    repo_dir = os.path.join(OUTPUT_DIR, repo_name.replace("/", "_"))
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
        print(f"[INFO] Created directory: {repo_dir}")

    metadata_file = os.path.join(repo_dir, "metadata.json")

    metadata = _load_metadata(metadata_file)
    existing_pr_numbers = set(metadata.get('processed_pr_numbers', []))
    processed_with_python = metadata.get('processed_with_python', 0)
    csv_file_counter = metadata.get('csv_file_counter', 0)
    lowest_pr_number = metadata.get('lowest_pr_number', None)

    print(f"[INFO] Starting from CSV file #{csv_file_counter + 1}")
    print(f"[INFO] Progress: {processed_with_python}/{TARGET_PRS_WITH_PYTHON}")
    if lowest_pr_number:
        print(f"[INFO] Will fetch PRs below #{lowest_pr_number}")

    remaining = TARGET_PRS_WITH_PYTHON - processed_with_python
    if remaining <= 0:
        print(f"[INFO] ✓ Target reached! ({processed_with_python}/{TARGET_PRS_WITH_PYTHON})")
        return

    df_rows = []
    skipped_no_python = 0

    while processed_with_python < TARGET_PRS_WITH_PYTHON:
        print(f"\n{'=' * 60}")
        print(f"Fetching PRs below #{lowest_pr_number if lowest_pr_number else 'latest'}")
        print(f"{'=' * 60}")

        pr_objs = fetch_pr_list_descending(repo_name, max_pr_number=lowest_pr_number, max_prs=FETCH_BATCH_SIZE)
        if not pr_objs:
            print("[INFO] No more PRs available")
            break

        new_pr_objs = [pr for pr in pr_objs if pr.number not in existing_pr_numbers]
        print(f"✓ {len(new_pr_objs)} new PRs to process")

        if not new_pr_objs:
            if pr_objs:
                lowest_pr_number = min(pr.number for pr in pr_objs) - 1
            continue

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pr = {
                executor.submit(fetch_single_pr, repo_name, pr_obj.number): pr_obj
                for pr_obj in new_pr_objs
            }

            for future in tqdm(as_completed(future_to_pr),
                               total=len(future_to_pr),
                               desc="Processing",
                               unit="PR"):
                pr_obj = future_to_pr[future]

                try:
                    pr_dict = future.result()
                    if pr_dict:
                        if not _has_python_files(pr_dict):
                            skipped_no_python += 1
                            existing_pr_numbers.add(pr_obj.number)
                            continue

                        df_rows.append(pr_dict)
                        existing_pr_numbers.add(pr_obj.number)
                        processed_with_python += 1

                        # Save batch when threshold reached
                        if len(df_rows) >= SAVE_EVERY:
                            csv_file_counter += 1
                            csv_filename = f"{repo_name.replace('/', '_')}_initial_{csv_file_counter}.csv"
                            csv_path = os.path.join(repo_dir, csv_filename)

                            rows_saved = _save_batch_to_separate_csv(csv_path, df_rows)

                            if rows_saved > 0:
                                # Update lowest_pr_number from current batch
                                pr_numbers_in_batch = [pr['number'] for pr in df_rows]
                                lowest_pr_number = min(pr_numbers_in_batch)

                                # Save metadata
                                _save_metadata(metadata_file, {
                                    'processed_pr_numbers': list(existing_pr_numbers),
                                    'processed_with_python': processed_with_python,
                                    'csv_file_counter': csv_file_counter,
                                    'lowest_pr_number': lowest_pr_number
                                })
                                print(f"✓ Saved {csv_filename} | Progress: {processed_with_python}/{TARGET_PRS_WITH_PYTHON}")
                                df_rows = []
                            else:
                                print(f"[ERROR] Save failed - NOT clearing buffer!")

                        if processed_with_python >= TARGET_PRS_WITH_PYTHON:
                            break

                except Exception as e:
                    print(f"[WARNING] Failed PR #{pr_obj.number}: {e}")

        # Save remaining rows if we have any
        if df_rows and processed_with_python < TARGET_PRS_WITH_PYTHON:
            csv_file_counter += 1
            csv_filename = f"{repo_name.replace('/', '_')}_initial_{csv_file_counter}.csv"
            csv_path = os.path.join(repo_dir, csv_filename)

            rows_saved = _save_batch_to_separate_csv(csv_path, df_rows)
            if rows_saved > 0:
                pr_numbers_in_batch = [pr['number'] for pr in df_rows]
                lowest_pr_number = min(pr_numbers_in_batch)

                _save_metadata(metadata_file, {
                    'processed_pr_numbers': list(existing_pr_numbers),
                    'processed_with_python': processed_with_python,
                    'csv_file_counter': csv_file_counter,
                    'lowest_pr_number': lowest_pr_number
                })
                print(f"✓ Saved {csv_filename} (final partial batch)")
                df_rows = []

        # Update lowest_pr_number for next iteration
        if existing_pr_numbers and not df_rows:
            current_min = min(existing_pr_numbers)
            if lowest_pr_number is None or current_min < lowest_pr_number:
                lowest_pr_number = current_min

        if processed_with_python >= TARGET_PRS_WITH_PYTHON:
            break

    print(f"\n{'=' * 60}")
    print(f"✓ FINISHED {repo_name}")
    print(f"Total CSV files created: {csv_file_counter}")
    print(f"Total with Python: {processed_with_python}")
    print(f"Skipped (no Python): {skipped_no_python}")
    print(f"{'=' * 60}")


def _save_batch_to_separate_csv(csv_path, df_rows):
    if not df_rows:
        return 0

    try:
        print(f"\n[SAVE] Creating new CSV with {len(df_rows)} PRs...")

        # Build dataframe
        try:
            batch_df = build_initial_pr_dataframe(df_rows)
        except Exception as e:
            print(f"[ERROR] build_initial_pr_dataframe crashed: {e}")
            import traceback
            traceback.print_exc()
            return 0

        if batch_df is None or len(batch_df) == 0:
            print(f"[ERROR] build_initial_pr_dataframe returned empty/None!")
            return 0

        if len(batch_df) != len(df_rows):
            print(f"[WARNING] Row mismatch: {len(df_rows)} input → {len(batch_df)} output")

        # Ensure all columns exist
        for col in COLUMNS:
            if col not in batch_df.columns:
                batch_df[col] = None

        batch_df = batch_df[COLUMNS]

        # Write to NEW CSV file
        with csv_lock:
            batch_df.to_csv(csv_path, mode='w', header=True, index=False)

        # Verify
        if os.path.exists(csv_path):
            verify_df = pd.read_csv(csv_path)
            rows_saved = len(verify_df)
            print(f"[SAVE] Created {os.path.basename(csv_path)} with {rows_saved} rows")
            return rows_saved
        else:
            print(f"[ERROR] File was not created: {csv_path}")
            return 0

    except Exception as e:
        print(f"[ERROR] _save_batch_to_separate_csv crashed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def _has_python_files(pr_dict):
    """Check if PR has Python files."""
    if not pr_dict.get('files_metrics'):
        return False
    for f in pr_dict['files_metrics']:
        if f.get("filename", "").endswith(".py"):
            return True
    return False


def _load_metadata(metadata_file):
    """Load metadata JSON."""
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_metadata(metadata_file, metadata):
    """Save metadata JSON."""
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save metadata: {e}")


if __name__ == "__main__":
    if not REPOS:
        print("[ERROR] No repositories defined")
        sys.exit(1)

    for repo_name in REPOS:
        try:
            process_repository(repo_name)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"[CRITICAL] Error in {repo_name}: {e}")
            import traceback
            traceback.print_exc()