# Releasing

Two artefacts with two DOIs: this repository, archived by Zenodo from a GitHub
release, and the replication dataset, uploaded to Zenodo by hand. They are
separate on purpose. The code is small, text, and changes; the data is 546 MB,
binary, and carries terms the code does not.

## 1. First push

The GitHub repository `gabrielkasmi/pv-registry-audit` already exists.

```bash
cd /path/to/source

git init -b main
git add -A
git status --short
```

**Read `git status` before committing.** Two checks, in this order.

```bash
git ls-files data/            # must print exactly: data/MANIFEST.csv
du -sh .git                   # sanity: tens of MB, not hundreds
```

If anything else under `data/` appears, stop and fix `.gitignore` first. A file
committed once stays in the history after deletion, and scrubbing it means
rewriting the history of a repository the paper cites. The `.gitignore` here
excludes `data/*` and re-includes the manifest; a bare `data/` line would break
that, and the file says why.

The repository should come to roughly 45 MB: 11 MB of code, 34 MB of figures.
No file exceeds 4 MB, so no Git LFS.

```bash
git commit -m "Analysis code for the registry audit"
git remote add origin git@github.com:gabrielkasmi/pv-registry-audit.git
git push -u origin main
```

## 2. Before the first release

Two things that are painful to retrofit.

**Turn on the Zenodo switch** at <https://zenodo.org/account/settings/github/>,
signing in with GitHub. Zenodo only archives releases published *after* the
switch is on, so doing this afterwards means deleting the release and
republishing it. The repository must be public for it to appear; click *Sync
now* if it does not.

**Decide the version.** `CITATION.cff` says `1.0.0`. Tag accordingly.

## 3. Release

```bash
git tag -a v1.0.0 -m "v1.0.0 — code accompanying the paper"
git push origin v1.0.0
```

Then create a GitHub Release from that tag. Zenodo archives it and mints a DOI.

Take the **concept DOI**, the one the record page describes as citing all
versions, not the version DOI. Adding the data DOI later, or fixing a typo in a
notebook, should not invalidate the citation in the paper.

## 4. The data deposit

Separate, manual, and gated on the operator's clearance.

```bash
cd /path/to/source
zip -r pv-registry-audit-data.zip data/
```

Upload to Zenodo as its own record, `upload_type: dataset`. In its metadata, set
a *related identifier* with relation **isSupplementTo** pointing at the code DOI,
and add the reciprocal **isSupplementedBy** on the code record. Without that
pair, a reader who finds one does not find the other.

The deposit carries `data/source/rte_derived/README.md`, which states the scope
of the operator agreement file by file. Point to it in the Zenodo description
rather than restating it.

## 5. Fill the placeholders

Once both DOIs exist:

| Where | What |
|---|---|
| `README.md` | `[DOI-DATA]` in the setup section |
| `REPRODUCE.md` | `[DOI-DATA]` in the setup section |
| `CITATION.cff` | add `doi:` for this repository, and `date-released:` |
| `.zenodo.json` | add the data DOI to `related_identifiers` |
| `joule_main.tex` | `[DOI-CODE]` and `[DOI-DATA]` in *Data and code availability* |

The bib entry for this repository does not exist yet either.

Then push once more, and tag `v1.0.1` if you want the archived version to carry
the filled-in DOIs. That is optional: the concept DOI already resolves to
whatever is latest.

## Checklist

- [ ] `git ls-files data/` returns only the manifest
- [ ] `.git` is tens of MB
- [ ] Repository public, Zenodo switch **on**
- [ ] Tag pushed, GitHub Release published, Zenodo record created
- [ ] Concept DOI, not version DOI, taken for the paper
- [ ] Data zipped and uploaded, once the operator has cleared it
- [ ] The two records cross-reference each other
- [ ] Placeholders filled in the five places above
