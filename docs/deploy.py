"""build the zensical site and push it to the ``nist-pages`` branch.

pages.nist.gov serves that branch at https://pages.nist.gov/striqt/ and requires
fast-forward commits, so the site is committed on top of the existing branch in a
temporary clone. The build runs through ``pixi run -e doc docs``, which regenerates
``docs/api`` first; the deploy stops when the regenerated pages differ from the
committed ones.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / 'site'
BRANCH = 'nist-pages'


def run(*args: str, cwd: Path, capture: bool = False) -> str:
    """run a command in `cwd` and return its stdout; a non-zero exit raises."""
    result = subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=capture
    )
    return result.stdout if capture else ''


def banner(text: str) -> None:
    line = '=' * 40
    print(f'{line}\n{text}\n{line}', flush=True)


def build() -> None:
    """generate the API pages and build the site; stale committed pages stop here."""
    banner('1. Generating the API pages and building the site')
    run('pixi', 'run', '-e', 'doc', 'docs', cwd=ROOT)
    stale = subprocess.run(
        ['git', 'diff', '--quiet', '--', 'docs/api'], cwd=ROOT, check=False
    )
    if stale.returncode != 0:
        sys.exit(
            'docs/api changed when regenerated; commit the regenerated pages and rerun.'
        )


def checkout_branch(remote_url: str, clone_dir: Path) -> None:
    """clone `BRANCH` from `remote_url` into `clone_dir`, or create it as an orphan.

    The tracked files are removed from the index and the working tree, so a page
    deleted from the site leaves the branch.
    """
    banner('2. Preparing temporary deployment environment')
    heads = run(
        'git', 'ls-remote', '--heads', remote_url, BRANCH, cwd=ROOT, capture=True
    )
    if BRANCH in heads:
        print(f"Cloning existing '{BRANCH}' branch...")
        run(
            'git', 'clone', '--depth', '1', '--branch', BRANCH, remote_url,
            str(clone_dir), cwd=ROOT,
        )  # fmt: skip
        run('git', 'rm', '-r', '-f', '-q', '--ignore-unmatch', '.', cwd=clone_dir)
    else:
        print(f"Branch '{BRANCH}' not found. Creating a new orphan branch...")
        run('git', 'clone', remote_url, str(clone_dir), cwd=ROOT)
        run('git', 'checkout', '--orphan', BRANCH, cwd=clone_dir)
        run('git', 'rm', '-r', '-f', '-q', '.', cwd=clone_dir)


def stage_site(clone_dir: Path) -> bool:
    """copy the site into `clone_dir` and stage it; return whether anything changed."""
    banner('3. Staging and pushing the site')
    shutil.copytree(SITE_DIR, clone_dir, dirs_exist_ok=True)
    # pages.nist.gov runs Jekyll on the branch unless this file exists
    (clone_dir / '.nojekyll').touch()
    run('git', 'add', '-A', cwd=clone_dir)
    staged = subprocess.run(
        ['git', 'diff', '--staged', '--quiet'], cwd=clone_dir, check=False
    )
    return staged.returncode != 0


def push(clone_dir: Path) -> None:
    stamp = run('date', '+%Y-%m-%d %H:%M:%S', cwd=clone_dir, capture=True).strip()
    run('git', 'commit', '-q', '-m', f'Deploy docs to {BRANCH}: {stamp}', cwd=clone_dir)
    print(f'Pushing changes to origin/{BRANCH}...')
    run('git', 'push', 'origin', BRANCH, cwd=clone_dir)
    banner(f'Success! Documentation deployed to {BRANCH}.')


def main() -> int:
    remote_url = run('git', 'remote', 'get-url', 'origin', cwd=ROOT, capture=True)
    build()
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / 'clone'
        checkout_branch(remote_url.strip(), clone_dir)
        if stage_site(clone_dir):
            push(clone_dir)
        else:
            print('No documentation changes detected. Nothing to push.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
