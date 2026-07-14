# Publishing Quasar — setup guide

Do these in order. Skipping ahead will make the PyPI form reject you.

---

## ⚠️ Before anything: check the name is free

Open: **https://pypi.org/project/quasar-context/**

- **404 Not Found** → the name is free. Continue.
- **A page loads** → the name is taken. Pick another (`quasar-rag`, `quasar-ctx`,
  `faithful-context`), then update `name = "..."` in `pyproject.toml` and use the
  new name everywhere below.

---

## Step 1 — Create the GitHub repo and push

```bash
cd quasar_pkg

git init
git add .
git commit -m "Quasar v0.1.0 — faithful context optimization"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
git push -u origin main
```

Write down two things exactly as they appear in your repo URL:

- **Your GitHub username** (or org name) → this is the **Owner**
- **Your repo name** → this is the **Repository name**

> The PyPI form needs these to match your real repo character-for-character.

---

## Step 2 — Create the GitHub Environment

In your repo on GitHub:

**Settings → Environments → New environment**

Name it exactly:

```
pypi
```

Optionally tick **Required reviewers** and add yourself. This means every release
waits for your explicit approval before it publishes — a real safety net, since
**PyPI version numbers are permanent and cannot be re-uploaded.**

Repeat once more to create a second environment named:

```
testpypi
```

---

## Step 3 — Register the Trusted Publisher on PyPI

Go to **pypi.org → Your projects → Publishing** (or the form you already have open).

Fill it in with **exactly** these values:

| Field | Value |
|---|---|
| **PyPI Project Name** | `quasar-context` |
| **Owner** | your GitHub username, e.g. `sebastiansabo` |
| **Repository name** | your repo name, e.g. `quasar` |
| **Workflow name** | `publish.yml` |
| **Environment name** | `pypi` |

### Common mistakes that break this

- ❌ **PyPI Project Name ≠ the name in `pyproject.toml`.** It is
  `quasar-context` (with the hyphen), not `Quasar`. They must match exactly.
- ❌ **Owner is not a made-up short code.** It's your literal GitHub username or
  organisation name.
- ❌ **Repository name is not your local folder name.** It's the repo name on
  GitHub.

---

## Step 4 — Rehearse on TestPyPI (strongly recommended)

Register a **second** trusted publisher, identical to the above, but at
**test.pypi.org**, with:

| Field | Value |
|---|---|
| Workflow name | `test-publish.yml` |
| Environment name | `testpypi` |

Then in GitHub: **Actions → Publish to TestPyPI → Run workflow**.

Verify it worked:

```bash
pip install -i https://test.pypi.org/simple/ quasar-context
python -c "from quasar import ContextOptimizer; print('works')"
```

If that installs and imports cleanly, you're safe to do the real thing.

---

## Step 5 — Publish for real

On GitHub: **Releases → Draft a new release**

- **Tag:** `v0.1.0`
- **Title:** `Quasar v0.1.0`
- Click **Publish release** (not "Save draft" — drafts don't trigger the workflow)

The `publish.yml` workflow fires:

1. Builds the package
2. **Runs the test suite** — the faithfulness contract must pass
3. Publishes to PyPI *only if the tests passed*

Then anyone can:

```bash
pip install quasar-context
```

---

## Releasing a new version later

1. Bump `version = "0.1.1"` in `pyproject.toml`
2. Commit and push
3. Draft a new release with tag `v0.1.1` → publish

**You can never reuse a version number.** If `0.1.0` ships broken, you fix it and
ship `0.1.1`. There is no overwriting.

---

## What the workflows do

| File | Trigger | What it does |
|---|---|---|
| `tests.yml` | every push + PR | Runs the test suite on Python 3.9 / 3.11 / 3.12 |
| `publish.yml` | GitHub Release published | Builds → tests → publishes to PyPI |
| `test-publish.yml` | manual (Actions tab) | Publishes to TestPyPI for rehearsal |

`publish.yml` will **refuse to publish if the tests fail.** That's deliberate: the
tests assert that critical values survive compression and that failures are
reported rather than hidden. If that contract breaks, nothing ships.
