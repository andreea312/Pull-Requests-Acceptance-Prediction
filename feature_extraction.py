import re
import pandas as pd
import ast
import numpy as np
from radon.raw import analyze as raw_analyze
from radon.complexity import cc_visit
from typing import Dict, Any
from tqdm import tqdm


class ASTMetricsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0
        self.func_defs = 0
        self.max_func_args = 0
        self.total_calls = 0
        self.total_if = 0
        self.total_loops = 0

    def generic_visit(self, node):
        old_depth = self.current_depth
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
        super().generic_visit(node)
        self.current_depth = old_depth

    def visit_FunctionDef(self, node):
        self.func_defs += 1
        num_args = len(node.args.args) + (1 if node.args.vararg else 0) + (1 if node.args.kwarg else 0)
        self.max_func_args = max(self.max_func_args, num_args)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.total_calls += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.total_if += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.total_loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.total_loops += 1
        self.generic_visit(node)


def compute_python_metrics(file_content: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(file_content)
        visitor = ASTMetricsVisitor()
        visitor.visit(tree)
        ast_metrics = {
            "max_nesting": visitor.max_depth,
            "func_count": visitor.func_defs,
            "max_args": visitor.max_func_args,
            "call_count": visitor.total_calls,
            "if_count": visitor.total_if,
            "loop_count": visitor.total_loops
        }
    except Exception:
        ast_metrics = {k: None for k in ["max_nesting", "func_count", "max_args", "call_count",
                                         "if_count", "loop_count"]}

    try:
        raw = raw_analyze(file_content)
        loc, lloc, sloc, comments, multi, blank = raw.loc, raw.lloc, raw.sloc, raw.comments, raw.multi, raw.blank
    except Exception:
        loc = lloc = sloc = comments = multi = blank = None

    try:
        cc_list = cc_visit(file_content)
        cc_scores = [c.complexity for c in cc_list] if cc_list else []
        avg_cc = float(np.mean(cc_scores)) if cc_scores else None
        max_cc = float(np.max(cc_scores)) if cc_scores else None
    except Exception:
        avg_cc = max_cc = None

    return {**ast_metrics,
            "avg_cc": avg_cc,
            "max_cc": max_cc,
            "loc": loc,
            "lloc": lloc,
            "sloc": sloc,
            "comments": comments,
            "multi_comments": multi,
            "blank": blank}


def compute_basic_metrics(content: str) -> Dict[str, Any]:
    lines = content.splitlines()
    loc = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comments = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*", "*")))
    sloc = loc - blank - comments
    return {"loc": loc, "sloc": sloc, "blank": blank, "comments": comments, "multi_comments": None,
            "max_nesting": None, "func_count": None, "max_args": None, "call_count": None, "if_count": None,
            "loop_count": None,
            "avg_cc": None, "max_cc": None}


def aggregate_file_metrics(pr: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["max_nesting", "func_count", "max_args", "call_count", "if_count", "loop_count",
            "avg_cc", "max_cc", "loc", "lloc", "sloc", "comments", "multi_comments", "blank"]

    agg = {}
    files_metrics = pr.get("files_metrics", []) or []

    for key in keys:
        values = [f.get(key) for f in files_metrics if f.get(key) is not None]
        if values:
            agg[f"min_{key}"] = min(values)
            agg[f"avg_{key}"] = float(np.mean(values))
            agg[f"max_{key}"] = max(values)
        else:
            agg[f"min_{key}"] = agg[f"avg_{key}"] = agg[f"max_{key}"] = None

    return agg


def build_initial_pr_dataframe(pr_list):
    df_rows = []

    for pr in tqdm(pr_list, desc="Building dataframe for PRs", unit="PR"):
        files_metrics = pr.get("files_metrics", [])

        # Process files that have content
        files_with_content = 0
        for f in files_metrics:
            fname = f.get("filename", "")
            content = f.get("content", "")
            if not content:
                continue  # skip empty files but continue processing PR

            files_with_content += 1
            if fname.endswith(".py"):
                f.update(compute_python_metrics(content))
            else:
                f.update(compute_basic_metrics(content))

        # Aggregate metrics (even if no files had content) ---
        agg_metrics = aggregate_file_metrics(pr)
        pr.update(agg_metrics)

        # PR-level features
        pr["title_length"] = len(pr.get("title", "")) if pr.get("title") else None
        pr["description_length"] = len(pr.get("body", "")) if pr.get("body") else None
        pr["files_with_content"] = files_with_content  # Track how many files had content

        # Semantic labels using regex
        def contains_keywords_regex(text, patterns):
            if not text:
                return 0
            text_lower = text.lower()
            return int(any(re.search(p, text_lower) for p in patterns))

        bugfix_patterns = [r'\bfix\b', r'\bbug\b', r'\berror\b', r'\bissue\b',
                           r'\bfixes\b', r'\bbugs\b', r'\berrors\b', r'\bissues\b',
                           r'\bfixing\b', r'\bfixed\b', r'\bproblem\b', r'\bpatch\b',
                           r'\bcorrect\b', r'\bresolve\b', r'\bresolved\b', r'\bhotfix\b']
        refactor_patterns = [r'\brefactor\b', r'\bcleanup\b', r'\brefactored\b', r'\bcleaning\b',
                             r'\brefactoring\b', r'\brewrite\b', r'\brestructured\b', r'\bmodularize\b']
        feature_patterns = [r'\badd\b', r'\bfeature\b', r'\bimplement\b', r'\bintroduce\b',
                            r'\bcreate\b', r'\bnew\b', r'\bupgrade\b', r'\benable\b', r'\bimprove\b']

        pr["is_bugfix"] = contains_keywords_regex(pr.get("title", ""), bugfix_patterns)
        pr["is_refactor"] = contains_keywords_regex(pr.get("title", ""), refactor_patterns)
        pr["is_feature"] = contains_keywords_regex(pr.get("title", ""), feature_patterns)

        # Remove raw body
        pr.pop("body", None)

        # Always append the PR, even if no files had content
        df_rows.append(pr)

    # Ensure consistent columns
    all_keys = set().union(*[pr.keys() for pr in df_rows])
    for pr in df_rows:
        for k in all_keys:
            if k not in pr:
                pr[k] = None

    df = pd.DataFrame(df_rows)

    print(f"[INFO] build_initial_pr_dataframe: Input {len(pr_list)} PRs → Output {len(df)} rows")

    return df