# Firebase Authentication setup

TruthLens uses Firebase Authentication in the React client and Firebase Admin SDK in FastAPI. The frontend obtains a Firebase ID token (a signed JWT) and sends it as an HTTP Bearer token. The API validates signature, issuer/project, expiry and revocation before protected media analysis.

## 1. Configure Firebase Authentication

1. Open the [Firebase Console](https://console.firebase.google.com/) and create/select the project for TruthLens.
2. In **Project settings → General → Your apps**, register a Web app and copy its Firebase web config.
3. In **Authentication → Sign-in method**, enable **Email/Password** and **Google**.
4. In **Authentication → Settings → Authorized domains**, make sure `localhost` and your deployed frontend domain are listed.
5. Use the same Firebase project ID for the frontend and backend.

## 2. Configure the React frontend

From `frontend/`, copy the example file:

```powershell
Copy-Item .env.example .env.local
```

Fill in all `VITE_FIREBASE_*` values from the Firebase Web app config. Keep `VITE_API_URL=http://127.0.0.1:8000` for local development, or set it to the deployed API URL.

Restart Vite after changing environment variables:

```powershell
npm install
npm run dev
```

Firebase web config values are intended for the browser. They are not substitutes for backend credentials; restrict the associated API key by application/referrer in Google Cloud where practical.

## 3. Configure Firebase Admin for FastAPI

In Firebase Console, open **Project settings → Service accounts** and generate a private service-account key for local development. Save it as:

`backend/firebase-service-account.json`

Copy `backend/.env.example` to `backend/.env` and set:

```dotenv
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-service-account.json
```

The service-account JSON is private server credentials. Never paste it into source code, frontend variables, issue comments, or GitHub. The repository ignores `backend/firebase-service-account*.json` and local `.env` files.

For deployment, prefer your hosting provider's secret manager. Set `FIREBASE_PROJECT_ID` and provide the full JSON through `FIREBASE_SERVICE_ACCOUNT_JSON`, or attach a service identity and use Application Default Credentials. Do not commit credentials.

Restart FastAPI after changing backend environment variables:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## 4. Verify authentication

1. Open the frontend and create an account with email/password or choose Google sign-in.
2. The API health endpoint remains public at `/health`. In Settings, choose **Test API connection**; this calls the protected `/api/me` endpoint with the current Firebase ID token.
3. A signed-in user can submit an investigation; the frontend sends `Authorization: Bearer <Firebase ID token>` to `POST /api/analyze`.
4. Calling `/api/me` or `/api/analyze` without a valid token returns HTTP 401. If backend Firebase credentials are missing, protected calls fail closed with HTTP 503.

## 5. Account and report behavior

- Firebase manages sign-in, password handling, Google OAuth and client session refresh.
- FastAPI verifies Firebase ID tokens with `check_revoked=True`; the API does not issue its own JWT signing keys.
- Case history is currently saved in browser local storage under the Firebase user UID. It is separated by account in that browser, but it is **not cloud-synced**, shared between devices, or a server-side database.
- The API currently protects the analysis endpoint and authenticated identity endpoint. It does not yet offer persistent server-side case records, team roles, tenant permissions or evidence ownership in a database.
- Use HTTPS for any non-local deployment. Restrict authorized domains and configure `TRUTHLENS_FRONTEND_ORIGINS` for the deployed frontend URL.
