-- TruthLens core evidence schema.
BEGIN;
CREATE TABLE IF NOT EXISTS public.truthlens_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firebase_uid TEXT NOT NULL,
    case_title TEXT NOT NULL,
    overall_status TEXT NOT NULL DEFAULT 'inconclusive',
    summary TEXT NOT NULL DEFAULT '',
    report_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS public.truthlens_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES public.truthlens_cases(id) ON DELETE CASCADE,
    firebase_uid TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK (media_type IN ('image','video','audio','document')),
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-fA-F]{64}$'),
    analysis_status TEXT NOT NULL,
    storage_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    observations JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_assessment JSONB NOT NULL DEFAULT '{}'::jsonb,
    video_analysis JSONB NOT NULL DEFAULT '{}'::jsonb,
    forensic_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    text_extraction JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (case_id, asset_id)
);
CREATE TABLE IF NOT EXISTS public.truthlens_evidence_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id UUID NOT NULL REFERENCES public.truthlens_cases(id) ON DELETE CASCADE,
    firebase_uid TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    previous_hash CHAR(64) NOT NULL,
    event_hash CHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE public.truthlens_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_evidence_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.truthlens_cases FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_assets FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_evidence_events FROM PUBLIC, anon, authenticated;
COMMIT;
