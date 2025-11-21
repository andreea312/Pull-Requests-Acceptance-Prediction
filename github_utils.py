import time
from github import Github, RateLimitExceededException, GithubException
from collections import defaultdict
from tqdm import tqdm
from config import GITHUB_TOKEN1, GITHUB_TOKEN2, GITHUB_TOKEN3, GITHUB_TOKEN4, GITHUB_TOKEN5

g1 = Github(GITHUB_TOKEN1) if GITHUB_TOKEN1 else Github()
g2 = Github(GITHUB_TOKEN2) if GITHUB_TOKEN2 else Github()
g3 = Github(GITHUB_TOKEN3) if GITHUB_TOKEN3 else Github()
g4 = Github(GITHUB_TOKEN4) if GITHUB_TOKEN4 else Github()
g5 = Github(GITHUB_TOKEN5) if GITHUB_TOKEN5 else Github()

TOKENS = [g5]
token_index = 0
author_pr_cache = defaultdict(list)


def _get_client():
    global token_index
    client = TOKENS[token_index]
    token_index = (token_index + 1) % len(TOKENS)
    return client


def _safe_github_call(func, *args, **kwargs):
    while True:
        client = _get_client()
        try:
            return func(client, *args, **kwargs)

        except RateLimitExceededException as e:
            reset = int(e.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 5)
            print(f"\n[RATE LIMIT] Token exhausted. Waiting {wait} seconds...")
            time.sleep(wait)

        except GithubException as e:
            if e.status == 403:
                reset = int(e.headers.get("X-RateLimit-Reset", 0))
                if reset > 0:
                    wait = max(reset - int(time.time()), 5)
                    print(f"\n[RATE LIMIT] 403 with reset header. Waiting {wait} seconds...")
                    time.sleep(wait)
                    continue
            raise e

        except Exception:
            time.sleep(2)


def fetch_pr_list(repo_full_name, max_prs=1000, state="closed", skip=0):
    repo = _safe_github_call(lambda c, repo: c.get_repo(repo), repo_full_name)
    pulls = repo.get_pulls(state=state, sort="created", direction="desc")
    all_prs = []
    items_per_page = 100
    start_page = skip // items_per_page
    skip_in_page = skip % items_per_page
    page = start_page
    skipped = 0
    pbar = tqdm(total=max_prs, desc=f"Fetching closed PRs", unit="PR")

    while len(all_prs) < max_prs:
        page_items = _safe_github_call(lambda c, pulls, p: pulls.get_page(p), pulls, page)
        if not page_items:
            break

        for pr in page_items:
            if page == start_page and skipped < skip_in_page:
                skipped += 1
                continue

            # Keep only closed unmerged PRs
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


def fetch_single_pr(repo_full_name, pr_number):
    repo = _safe_github_call(lambda c, name: c.get_repo(name), repo_full_name)
    pr = _safe_github_call(lambda c, r, num: r.get_pull(num), repo, pr_number)

    pr_dict = {
        "number": pr.number,
        "title": pr.title or None,
        "body": pr.body or None,
        "created_at": pr.created_at.isoformat() if pr.created_at else None,
        "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "additions": pr.additions or -1,
        "deletions": pr.deletions or -1,
        "changed_files": pr.changed_files or -1,
        "commits": pr.commits or -1,
        "author": pr.user.login if pr.user else None
    }

    # Comments
    try:
        comments = _safe_github_call(lambda c, pr: list(pr.get_issue_comments()), pr)
        pr_dict["comments_list"] = [
            {
                "user": c.user.login if c.user else None,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in comments
        ]
        pr_dict["reviewers_list"] = list({c.user.login for c in comments if c.user})
    except Exception:
        pr_dict["comments_list"] = []
        pr_dict["reviewers_list"] = []

    # Commits
    try:
        commits = _safe_github_call(lambda c, pr: list(pr.get_commits()), pr)
        pr_dict["commits_list"] = [
            {
                "author": c.author.login if c.author else None,
                "timestamp": c.commit.committer.date.isoformat()
                if c.commit and c.commit.committer else None
            }
            for c in commits
        ]
    except Exception:
        pr_dict["commits_list"] = None

    # Files + full contents
    pr_dict["files_metrics"] = []
    try:
        commits = _safe_github_call(lambda c, pr: list(pr.get_commits()), pr)
        last_sha = commits[-1].sha if commits else pr.head.sha
        files = _safe_github_call(lambda c, pr: list(pr.get_files()), pr)
        for f in files:
            try:
                content_file = _safe_github_call(
                    lambda c, repo, filename, ref: repo.get_contents(filename, ref=ref),
                    repo, f.filename, last_sha
                )
                content = content_file.decoded_content.decode("utf-8")
            except Exception:
                content = None

            pr_dict["files_metrics"].append({"filename": f.filename, "content": content})

    except Exception:
        pr_dict["files_metrics"] = None

    return pr_dict
