# Supabase setup for TruthLens

The Supabase project `truthlens-db` has the core schema and a private Storage bucket provisioned through the migrations in `supabase/migrations/`.

## Provisioned resources

- `public.truthlens_cases`: per-Firebase-user case summary and complete JSON report.
- `public.truthlens_assets`: per-evidence metadata, SHA-256, modality, model findings and extracted text.
- `public.truthlens_evidence_events`: hash-linked case events.
- Private Storage bucket `truthlens-evidence`: 40 MB maximum per object and an allow-list of supported media/document MIME types.

## Backend cloud mirror

The API continues to authenticate users with Firebase. If both environment variables below are present on the backend, each completed case is mirrored to Supabase through its REST API:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Copy the values from the Supabase project's API settings into the backend deployment's secret environment variables. Never put the service-role key in `frontend/.env`, a `VITE_*` variable, source code, browser storage or GitHub.

The server uses the verified Firebase UID as `firebase_uid`; do not trust a UID supplied by the browser. The three tables have RLS enabled and intentionally have no client policies because Supabase Auth is not the app's identity provider. This denies direct anon/authenticated client access. A trusted backend using the service role bypasses RLS, so the API must continue enforcing Firebase report ownership.

## Current boundary

The cloud mirror currently stores case JSON, asset analysis metadata and hash-linked events. Original uploaded media files are still processed from temporary files and are not uploaded to the Storage bucket. The bucket is provisioned for a future consent-based original-evidence retention workflow; it is not yet wired into uploads or downloads.

The local SQLite database remains the backend's operational report/activity store. If Supabase credentials are absent, the app reports `not_configured` and uses local persistence. If credentials are configured but a cloud write fails, the analysis request returns a service error rather than claiming cloud sync succeeded.

## Validation

After setting the backend secrets, restart/redeploy the API and inspect Settings → Optional integration status. A `configured` Supabase persistence status only confirms both variables are present. Run an authenticated test case and verify that the case, assets and evidence events appear in the three Supabase tables.
