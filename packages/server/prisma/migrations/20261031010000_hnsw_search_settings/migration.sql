-- RF hnsw-search-settings: searches filter by project and user in the same SQL as the vector ordering
-- (packages/server/src/engine/memory/search.ts), and HNSW hands each query only hnsw.ef_search
-- candidates (default 40) before the WHERE runs, so a small project or user in a big table comes back
-- thin or empty. Raise the candidate budget for this database; on pgvector 0.8+ also turn on
-- strict-order iterative scan, which keeps the index scan and honours the filter. ALTER DATABASE ...
-- SET is idempotent, so running this twice changes nothing.
DO $$
DECLARE
  v text;
BEGIN
  EXECUTE format('ALTER DATABASE %I SET hnsw.ef_search = 100', current_database());
  SELECT extversion INTO v FROM pg_extension WHERE extname = 'vector';
  IF v IS NOT NULL AND string_to_array(v, '.')::int[] >= ARRAY[0, 8] THEN
    EXECUTE format('ALTER DATABASE %I SET hnsw.iterative_scan = strict_order', current_database());
  END IF;
END $$;
