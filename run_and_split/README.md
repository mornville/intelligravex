# repo-split

A small CLI to clone local repos into named “splits” and manage them for Codex, with optional Docker containers.

## Install

```bash
go install .
# or
# go build -o /usr/local/bin/repo-split
```

## Quick start

```bash
# Create a named split (defaults to the main repo)
repo-split -name task-a

# List splits
repo-split list

# Start Codex in a split
repo-split codex -id abc
```

## Defaults

- Source repo: `/Users/elidoruiz/dev/repos/gh-candorverse`
- Destination base: `./splits`
- Default Docker image: `repo-split-codex:latest`

## Common commands

```bash
# Clone
repo-split -name task-a
repo-split clone -name task-b -branch main

# List
repo-split list

# Remove by name
repo-split remove -name task-a

# Prune all
repo-split prune

# Resume a Codex session
repo-split codex -id abc -resume <session>
repo-split resume -id abc -session <session>
```

## Docker container support

The `container` command runs Codex inside a container with the split mounted at `/workspace`.

```bash
# Build image if missing + run Codex
repo-split container -id abc

# Open a shell in the same container
repo-split container -id abc -shell

# Stop/remove the container
repo-split container -id abc -stop
```

The container can pass through your environment and mount common credentials:

- `~/.codex` (for Codex auth)
- `~/.aws` (AWS CLI)
- `~/.pgpass` (Postgres)

### Prebuilt image

A Dockerfile is provided so you can pre-build:

```bash
cd /Users/elidoruiz/dev/tools/run_and_split
docker build -f Dockerfile.codex -t repo-split-codex:latest .
```

If the default image is missing, `repo-split container` will build it automatically.

## Auth helper

```bash
# Ensure Codex uses file-based auth and check cache
repo-split auth-sync
```

If `~/.codex/auth.json` is missing, run `codex login` on the host.

## Config

You can create a `config.json` to override defaults:

```json
{
  "source_repo": "/Users/elidoruiz/dev/repos/gh-candorverse",
  "dest_base": "/Users/elidoruiz/dev/tools/run_and_split/splits",
  "remote_url": "https://github.com/owner/repo.git",
  "branch": "main",
  "recurse_submodules": true,
  "use_local": true
}
```

## Notes

- `origin` is set after cloning. If `remote_url` is not provided, it copies `origin` from the source repo.
- IDs are short (3 characters) and shown in `repo-split list`.
- Use `repo-split restart` to rebuild the image and recreate all containers after updates.
