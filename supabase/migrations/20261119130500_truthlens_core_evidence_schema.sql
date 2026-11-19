create extension if not exists pgcrypto;

create table if not exists public.truthlens_cases (
  id uuid primary key default gen_random_uuid(),
  firebase_uid text not null,
  case_title text not null,
  overall_status text not null default 'inconclusive',
  summary text not null default '',
  report_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists truthlens_cases_owner_created_idx on public.truthlens_cases (firebase_uid, created_at desc);

create table if not exists public.truthlens_assets (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.truthlens_cases(id) on delete cascade,
  firebase_uid text not null,
  asset_id text not null,
  filename text not null,
  media_type text not null check (media_type in ('image','video','audio','document')),
  size_bytes bigint not null check (size_bytes >= 0),
  sha256 char(64) not null check (sha256 ~ '^[0-9a-fA-F]{64}$'),
  analysis_status text not null,
  storage_path text,
  metadata jsonb not null default '{}'::jsonb,
  observations jsonb not null default '[]'::jsonb,
  model_assessment jsonb not null default '{}'::jsonb,
  video_analysis jsonb not null default '{}'::jsonb,
  forensic_parameters jsonb not null default '{}'::jsonb,
  provenance jsonb not null default '{}'::jsonb,
  text_extraction jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (case_id, asset_id)
);
create index if not exists truthlens_assets_owner_case_idx on public.truthlens_assets (firebase_uid, case_id);
create index if not exists truthlens_assets_sha256_idx on public.truthlens_assets (sha256);

create table if not exists public.truthlens_evidence_events (
  id bigint generated always as identity primary key,
  case_id uuid not null references public.truthlens_cases(id) on delete cascade,
  firebase_uid text not null,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  previous_hash char(64) not null,
  event_hash char(64) not null unique,
  created_at timestamptz not null default now()
);
create index if not exists truthlens_events_owner_case_idx on public.truthlens_evidence_events (firebase_uid, case_id, id);

alter table public.truthlens_cases enable row level security;
alter table public.truthlens_assets enable row level security;
alter table public.truthlens_evidence_events enable row level security;

-- No client-facing policies are intentionally created: the app currently authenticates
-- with Firebase. Only a trusted backend using the Supabase service role may access these
-- tables until Firebase-to-Supabase JWT verification is implemented.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('truthlens-evidence','truthlens-evidence',false,41943040,array[
'image/jpeg','image/png','image/webp','image/bmp','image/tiff',
'video/mp4','video/quicktime','video/x-msvideo','video/x-matroska','video/webm',
'audio/wav','audio/flac','audio/ogg','text/plain','text/markdown','text/csv',
'application/pdf','application/vnd.openxmlformats-officedocument.wordprocessingml.document'
])
on conflict (id) do update set public=false,file_size_limit=excluded.file_size_limit,allowed_mime_types=excluded.allowed_mime_types;
