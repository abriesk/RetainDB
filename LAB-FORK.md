# About this fork

This fork of [RetainDB](https://github.com/RetainDB/RetainDB) keeps a self-hosted server running. It starts at upstream
`2ad2cc4` (2026-06-12, "Merge pull request #2 ... improve-retaindb-local"). On top of that it adds the fixes the self-hosted
(OSS) setup needed to build, migrate, ingest and search. Before this fork they were applied as patches to the built bundle; here
each one is a source commit, one defect per commit, with the reason in the commit message. The fixes live on branch `lab`;
`main` follows upstream.

## The fixes

| commit subject | what was wrong |
|---|---|
| prisma.$use | The pinned Prisma (6.19.3, from the lockfile) no longer has `$use`, so the server would not start. The BigInt-to-string middleware is replaced by `BigInt.prototype.toJSON`, which gives the same JSON output. Note: inside the server, query results now stay `bigint` until they are serialized. The current readers convert them with `Number()` |
| bare-hits | The search route read each vector hit as `{memory, similarity, relations}`, but `searchMemories` returns bare memories. The hits are now wrapped into that shape |
| ingest-offset | The ingest-session validator refused timestamps with a `+00:00` offset; only `Z` was accepted |
| org-cast | `org_id ::uuid` in the ingestion_jobs INSERT: the column is VARCHAR and the OSS org is the literal `default` |
| job-cast-select / job-cast-docs | `::uuid` casts in two lookups on VARCHAR columns (ingestion_jobs.id, ingestion_documents.job_id) |
| msg-conversation-id | `conversationId` in the messages `createMany`: the Message model has no such column |
| ingest-errors-logged | Two catch blocks in `ingestSession` pushed to an errors array nobody reads. They now also log to the server log |
| build-modules | Docker: `pnpm install --node-linker=hoisted`, so the copied `node_modules` alone resolves the server's dependencies, including `.prisma/client` |
| build-prisma | Docker: migrations run the Prisma CLI the lockfile resolves (6.19.3, on PATH). `npx` used to fetch Prisma 7, which rejects this schema (P1012). Telemetry is off |

## Verified

Each commit was built and run on a fresh stack. Migrations applied. A memory was added and then found by a paraphrased search.
A session ingest with `+00:00` timestamps completed, and its messages were stored. There were no 5xx responses.

**Not yet exercised live:** the vector-search path of `bare-hits`. The fresh stack had no embeddings, so only the
lexical path ran. The change is reviewed from source only.

## Upstream

None of these fixes has been sent upstream yet. Each one is small and self-contained, so they can be offered one by one.

## Credits

The defects were found, and the fixes first written as patches against the built bundle, by [Claude Code](https://claude.com/claude-code)
(Anthropic) while it ran a self-hosted trial of RetainDB. The patches were ported to source commits by the Hermes agent, and each
commit was re-checked by a second Claude session before it was published. The commit messages carry a `Co-Authored-By` trailer
for Claude.
