-- Stable report identifier enables Firebase UID scoped report retrieval.
BEGIN;
ALTER TABLE public.truthlens_cases ADD COLUMN IF NOT EXISTS report_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS truthlens_cases_owner_report_id_key
    ON public.truthlens_cases (firebase_uid, report_id) WHERE report_id IS NOT NULL;
COMMIT;
