-- Owner and chain lookup indexes for TruthLens core tables.
BEGIN;
CREATE INDEX IF NOT EXISTS truthlens_cases_owner_created_idx
    ON public.truthlens_cases (firebase_uid, created_at DESC);
CREATE INDEX IF NOT EXISTS truthlens_assets_owner_case_idx
    ON public.truthlens_assets (firebase_uid, case_id);
CREATE INDEX IF NOT EXISTS truthlens_assets_sha256_idx
    ON public.truthlens_assets (sha256);
CREATE INDEX IF NOT EXISTS truthlens_events_case_id_idx
    ON public.truthlens_evidence_events (case_id);
CREATE INDEX IF NOT EXISTS truthlens_events_owner_case_idx
    ON public.truthlens_evidence_events (firebase_uid, case_id, id);
COMMIT;
