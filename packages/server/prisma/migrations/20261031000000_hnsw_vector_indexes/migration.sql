-- RT0b7: upstream builds its vector indexes as ivfflat (lists = 100) on EMPTY tables and never sets ivfflat.probes,
-- so recall falls as memories grow (Margo TD-001). Swap them for HNSW (pgvector defaults m 16, ef_construction 64).
-- Same index names, so prisma's migration history is untouched. Idempotent: an index already hnsw is left alone;
-- a table or column that does not exist (entities) is skipped.
DO $$
DECLARE
  t text;
  idx text;
BEGIN
  FOREACH t IN ARRAY ARRAY['memories', 'chunks', 'entities'] LOOP
    idx := t || '_embedding_idx';
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = t AND column_name = 'embedding')
       AND NOT EXISTS (SELECT 1 FROM pg_indexes
                       WHERE schemaname = 'public' AND indexname = idx AND indexdef ILIKE '%USING hnsw%') THEN
      EXECUTE format('DROP INDEX IF EXISTS public.%I', idx);
      EXECUTE format('CREATE INDEX %I ON public.%I USING hnsw (embedding vector_cosine_ops)', idx, t);
    END IF;
  END LOOP;
END $$;
