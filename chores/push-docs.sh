#!/bin/bash
# Build the zensical site and push it to the `nist-pages` branch, which pages.nist.gov
# serves at https://pages.nist.gov/striqt/. Commits on that branch must stay
# fast-forwardable, so the script commits on top of the existing branch.
set -e

BUILD_DIR="site"
BRANCH="nist-pages"

REMOTE_URL=$(git remote get-url origin)

echo "========================================"
echo "1. Generating the API pages and building the site"
echo "========================================"
# the `docs` task regenerates docs/api/**/*.md from the public namespaces, then runs
# `zensical build --clean --strict`
pixi run -e doc docs

if ! git diff --quiet -- docs/api; then
    echo "docs/api changed when regenerated; commit the regenerated pages and rerun." >&2
    exit 1
fi

echo "========================================"
echo "2. Preparing temporary deployment environment"
echo "========================================"
TEMP_DIR=$(mktemp -d)
trap "rm -rf '$TEMP_DIR'" EXIT

if git ls-remote --heads "$REMOTE_URL" "$BRANCH" | grep -q "$BRANCH"; then
    echo "Cloning existing '$BRANCH' branch..."
    git clone --depth 1 --branch "$BRANCH" "$REMOTE_URL" "$TEMP_DIR"
    cd "$TEMP_DIR"
    # remove the tracked files so that deleted pages leave the branch
    git rm -rf . --ignore-unmatch > /dev/null
else
    echo "Branch '$BRANCH' not found. Creating a new orphan branch..."
    git clone "$REMOTE_URL" "$TEMP_DIR"
    cd "$TEMP_DIR"
    git checkout --orphan "$BRANCH"
    git rm -rf . > /dev/null
fi

echo "========================================"
echo "3. Staging and pushing the site"
echo "========================================"
cp -a "$OLDPWD/$BUILD_DIR/." .

# pages.nist.gov runs Jekyll on the branch unless this file exists
touch .nojekyll

git add -A

if git diff --staged --quiet; then
    echo "No documentation changes detected. Nothing to push."
else
    git commit -m "Deploy docs to $BRANCH: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Pushing changes to origin/$BRANCH..."
    git push origin "$BRANCH"
    echo "========================================"
    echo "Success! Documentation deployed to $BRANCH."
    echo "========================================"
fi
