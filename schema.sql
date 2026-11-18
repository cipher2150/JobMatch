-- Idempotent: the app runs this on startup.
create extension if not exists vector;

create table if not exists jobs (
  id text primary key,            -- "{source}:{source_id}"
  source text not null,
  title text, company text, location text, url text,
  body text not null,
  content_hash text not null,     -- sha256 of title+body
  embedding vector(1536),
  posted_at timestamptz,
  first_seen timestamptz default now(),
  last_seen timestamptz default now(),
  active boolean default true
);
-- Partial: every query filters on active, and HNSW filters after the scan, so inactive rows would eat the LIMIT.
create index if not exists jobs_embedding_active_idx on jobs using hnsw (embedding vector_cosine_ops) where active;

create table if not exists feedback (
  id bigserial primary key,
  user_key text not null,         -- random uuid in a cookie, no accounts
  job_id text references jobs(id),
  verdict text check (verdict in ('up','down')),
  created_at timestamptz default now()
);
