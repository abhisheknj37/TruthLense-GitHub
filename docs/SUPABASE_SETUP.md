# TruthLens Supabase Setup

## What is stored in Supabase

- `truthlens_cases`: full report JSON, case summary, Firebase owner UID, report ID and soft-delete timestamp.
- `truthlens_assets`: one record per uploaded image, audio or video, including SHA-256, forensic measurements, model outputs and private Storage object path.
- `truthlens_evidence_events`: hash-linked chain-of-custody events for each case. These remain after a case is soft-deleted.
- `truthlens_activity`: user-scoped investigation activity and dashboard summaries.
- `truthlens_heatmaps`: persisted PNG attribution overlays.
- Private Storage bucket `truthlens-evidence`: original evidence files, with a 40 MB object limit.

Firebase Authentication remains the identity provider. FastAPI verifies Firebase ID tokens and scopes every database query to that Firebase UID.

## Backend environment

Set these values in `backend/.env` for local development or as backend deployment secrets:

- `DATABASE_URL`: Supabase PostgreSQL **Session Pooler** URI (port 5432) with `sslmode=require`.
- `SUPABASE_URL`: `https://udbeldmgbzptuoenhbmn.supabase.co` for this project.
- `SUPABASE_SECRET_KEY`: current server-only API secret key (`sb_secret_...`) from Supabase API Keys. Never put it in frontend variables, source code or GitHub. The backend also accepts the legacy `SUPABASE_SERVICE_ROLE_KEY` during migration.
- `SUPABASE_STORAGE_BUCKET=truthlens-evidence`.

Never send database passwords or service-role keys in chat. If storage variables are absent, reports and activity can still use PostgreSQL, but original-media retention is shown as not configured.

## Connect and run

1. In Supabase Dashboard, choose **Connect** and copy **Session pooler** (port 5432). Add `sslmode=require` and place it in `DATABASE_URL`.
2. Add the private Storage URL and `SUPABASE_SECRET_KEY` to backend secrets only. Prefer the current `sb_secret_...` key; the API request sends it in the `apikey` header, not as a Bearer JWT.
3. Install dependencies with `python -m pip install -r requirements.txt` from `backend/`.
4. Restart FastAPI and check `GET /api/capabilities` for database and original-storage status.

## Security

- Every application table has RLS enabled; public `anon` and `authenticated` roles have no table privileges. Browser clients do not query case tables.
- FastAPI connects using a backend-only PostgreSQL connection string and separately verifies Firebase ID tokens. Every query includes the current Firebase UID. Database-owner connections bypass RLS, so the API's Firebase checks and explicit owner filters are essential.
- Original evidence is stored in a private bucket and streamed only through an authenticated backend route that first verifies case ownership.
- The service-role key is never returned by any API and must never be exposed to the frontend.
- Deleting a case soft-deletes its report and removes heatmaps/original Storage objects while retaining event-chain records and asset hashes for audit.
- Hash-linked events provide tamper-evident sequencing within the database, not externally anchored trusted timestamps or protection against full database replacement.

## Local fallback and existing data

Without `DATABASE_URL`, local development and CI continue to use SQLite. Existing SQLite data is not imported automatically; export it separately before switching production workloads.
