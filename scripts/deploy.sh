#!/usr/bin/env bash

# ref: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

if ! git diff-index --quiet HEAD --; then
    echo "dirty working tree; please clean or commit changes"
    exit 1
fi

if ! current="$(git describe --exact-match --tags HEAD 2> /dev/null)"; then
    echo "current revision not tagged; please deploy from a tagged revision"
    exit 1
fi

latest="$(git describe --tags "$(git rev-list --tags --max-count=1)")"

if [[ "$current" != "$latest" ]]; then
    echo "current revision is not the latest version; please deploy from latest version"
    exit 1
fi

# Authenticate via ~/.pypirc, a TWINE_USERNAME/TWINE_PASSWORD env pair, or the
# interactive prompt (twine supports all three).
twine upload dist/*

git push --tags
