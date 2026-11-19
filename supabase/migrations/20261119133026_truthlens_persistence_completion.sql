-- Complete TruthLens persistence schema, private original-media bucket and API lockdown.
BEGIN;
ALTER TABLE public.truthlens_cases ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS public.truthlens_activity (
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 case_id UUID NOT NULL REFERENCES public.truthlens_cases(id) ON DELETE CASCADE,
 firebase_uid TEXT NOT NULL,
 event_type TEXT NOT NULL,
 case_title TEXT NOT NULL,
 overall_status TEXT NOT NULL,
 evidence_count INTEGER NOT NULL CHECK (evidence_count >= 0),
 verified_count INTEGER NOT NULL CHECK (verified_count >= 0),
 evidence_json JSONB NOT NULL DEFAULT '[]'::jsonb,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 CONSTRAINT truthlens_activity_case_id_key UNIQUE (case_id)
);
CREATE INDEX IF NOT EXISTS truthlens_activity_owner_created_idx
    ON public.truthlens_activity (firebase_uid, created_at DESC);
CREATE TABLE IF NOT EXISTS public.truthlens_heatmaps (
 case_id UUID NOT NULL REFERENCES public.truthlens_cases(id) ON DELETE CASCADE,
 firebase_uid TEXT NOT NULL,
 asset_id TEXT NOT NULL,
 face_index INTEGER NOT NULL CHECK (face_index >= 0),
 image_data BYTEA NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY (case_id, asset_id, face_index)
);
CREATE INDEX IF NOT EXISTS truthlens_heatmaps_owner_case_idx
    ON public.truthlens_heatmaps (firebase_uid, case_id);
ALTER TABLE public.truthlens_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_evidence_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.truthlens_heatmaps ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.truthlens_cases FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_assets FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_evidence_events FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_activity FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.truthlens_heatmaps FROM PUBLIC, anon, authenticated;
DO $$ DECLARE seq_name TEXT; BEGIN
 seq_name := pg_get_serial_sequence('public.truthlens_evidence_events', 'id');
 IF seq_name IS NOT NULL THEN
   EXECUTE format('REVOKE ALL ON SEQUENCE %s FROM PUBLIC, anon, authenticated', seq_name);
 END IF;
 IF to_regprocedure('public.rls_auto_enable()') IS NOT NULL THEN
   REVOKE EXECUTE ON FUNCTION public.rls_auto_enable() FROM PUBLIC, anon, authenticated;
 END IF;
END $$;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON TABLES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC, anon, authenticated;
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('truthlens-evidence','truthlens-evidence',FALSE,41943040,
 ARRAY['image/jpeg','image/png','image/webp','image/bmp','image/tiff','video/mp4','video/quicktime','video/x-msvideo','video/x-matroska','video/webm','audio/wav','audio/flac','audio/ogg','text/plain','text/markdown','text/csv','application/pdf','application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, public=FALSE,
 file_size_limit=EXCLUDED.file_size_limit, allowed_mime_types=EXCLUDED.allowed_mime_types;
COMMIT;
