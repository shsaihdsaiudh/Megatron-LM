"""Export GitHub issue/PR discussions to markdown using the gh CLI.

Usage: python3 export_github.py OWNER/REPO OUT_DIR NUMBER...
"""

import json
import subprocess
import sys
from pathlib import Path


def gh(path, paginate=False):
    cmd = ["gh", "api"] + (["--paginate", "--slurp"] if paginate else []) + [path]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    if paginate:
        data = [item for page in data for item in page]
    return data


def export(repo, out_dir, number):
    issue = gh(f"repos/{repo}/issues/{number}")
    is_pr = "pull_request" in issue
    lines = [
        f"# {'PR' if is_pr else 'Issue'} #{number}: {issue['title']}\n",
        f"- URL: {issue['html_url']}",
        f"- Author: @{issue['user']['login']}  State: {issue['state']}",
    ]
    if is_pr:
        pr = gh(f"repos/{repo}/pulls/{number}")
        lines.append(f"- Merged: {pr.get('merged_at')}  Head ref on dev: refs/remotes/pr/{number}")
    lines += [f"- Created: {issue['created_at']}", "", "## Body", "", issue.get("body") or "", ""]

    lines += ["## Conversation", ""]
    for c in gh(f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True):
        lines += [f"### @{c['user']['login']} {c['created_at']}", "", c["body"] or "", ""]

    if is_pr:
        reviews = [r for r in gh(f"repos/{repo}/pulls/{number}/reviews?per_page=100", paginate=True) if r.get("body")]
        if reviews:
            lines += ["## Review summaries", ""]
            for r in reviews:
                lines += [f"### @{r['user']['login']} {r['state']} {r.get('submitted_at')}", "", r["body"], ""]
        inline = gh(f"repos/{repo}/pulls/{number}/comments?per_page=100", paginate=True)
        if inline:
            lines += ["## Inline review comments", ""]
            for c in inline:
                line = c.get("line") or c.get("original_line")
                lines += [
                    f"### @{c['user']['login']} {c['created_at']} `{c['path']}`:{line}",
                    "",
                    "```diff",
                    c["diff_hunk"][-600:],
                    "```",
                    "",
                    c["body"] or "",
                    "",
                ]

    name = f"{'pr' if is_pr else 'issue'}-{number}.md"
    (Path(out_dir) / name).write_text("\n".join(lines))
    print(name)


if __name__ == "__main__":
    repo, out_dir, *numbers = sys.argv[1:]
    for n in numbers:
        export(repo, out_dir, n)
