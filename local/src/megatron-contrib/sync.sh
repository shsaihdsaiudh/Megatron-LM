#!/usr/bin/env bash
# Run in WSL (the dev machine cannot reach GitHub).
# 1. Fetch NVIDIA/Megatron-LM main plus selected PR heads, push them to the dev-machine clone as
#    refs/remotes/origin/main and refs/remotes/pr/<N>.
# 2. Re-export selected issue/PR discussions to markdown and copy them to the dev machine.
#
# Usage: ~/src/megatron-contrib/sync.sh [PR or issue numbers...]
#   Numbers default to the ones tracked below. Each number is treated as a PR if it is one.
set -euo pipefail

REPO=NVIDIA/Megatron-LM
LOCAL_CLONE=~/src/Megatron-LM
HERE=$(cd "$(dirname "$0")" && pwd)
DEV_DIR=/planning-users/yangyeyuan/megatron-contrib
NUMS=("$@")
[ ${#NUMS[@]} -eq 0 ] && NUMS=(5616 6714 7814 5985 6973 8395 8787 9105)

quiet() { grep -v -E "post-quantum|store now|upgraded" || true; }

cd "$LOCAL_CLONE"
git fetch -q origin
for n in "${NUMS[@]}"; do
  if gh api "repos/$REPO/pulls/$n" --silent 2>/dev/null; then
    git fetch -q origin "+pull/$n/head:refs/remotes/pr/$n"
  fi
done
git push -q -f dev 'refs/remotes/origin/main:refs/remotes/origin/main' 'refs/remotes/pr/*:refs/remotes/pr/*' 2>&1 | quiet

mkdir -p "$HERE/github"
python3 "$HERE/export_github.py" "$REPO" "$HERE/github" "${NUMS[@]}"
ssh cloudml "mkdir -p $DEV_DIR/github" 2>&1 | quiet
# -O: legacy scp protocol; SFTP mode fails on JuiceFS "setstat" after copying.
scp -O -q "$HERE"/github/*.md "cloudml:$DEV_DIR/github/" 2>&1 | quiet
echo "synced main=$(git rev-parse --short origin/main) and ${NUMS[*]} to dev"
