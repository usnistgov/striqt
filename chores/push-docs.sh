#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

DOC_SRC="doc"
BUILD_DIR="doc/html"
BRANCH="nist-pages"

# Get the remote URL of the current repository
REMOTE_URL=$(git remote get-url origin)

echo "========================================"
echo "1. Building Sphinx documentation"
echo "========================================"
sphinx-build -E -a -b html "$DOC_SRC" "$BUILD_DIR"

echo "========================================"
echo "2. Preparing temporary deployment environment"
echo "========================================"
# Create a temporary directory that will automatically be deleted when the script exits
TEMP_DIR=$(mktemp -d)
trap "rm -rf '$TEMP_DIR'" EXIT

# Check if the branch already exists on the remote
if git ls-remote --heads "$REMOTE_URL" "$BRANCH" | grep -q "$BRANCH"; then
    echo "Cloning existing '$BRANCH' branch..."
    # Clone only the target branch, with a depth of 1 to keep it lightning fast
    git clone --depth 1 --branch "$BRANCH" "$REMOTE_URL" "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Remove all tracked files to ensure deleted doc pages are actually removed from the branch
    git rm -rf . --ignore-unmatch > /dev/null
else
    echo "Branch '$BRANCH' not found. Creating a new orphan branch..."
    git clone "$REMOTE_URL" "$TEMP_DIR"
    cd "$TEMP_DIR"
    git checkout --orphan "$BRANCH"
    git rm -rf . > /dev/null
fi

echo "========================================"
echo "3. Staging and pushing documentation"
echo "========================================"
# Copy the newly built HTML files into the temporary repo (. copies hidden files too)
cp -a "$OLDPWD/$BUILD_DIR/." .

# Create a .nojekyll file. This is crucial! 
# Sphinx outputs directories starting with an underscore (like _static), 
# which standard Pages servers will ignore unless this file exists.
touch .nojekyll

git add -A

# Check if there are actually any changes to commit
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
