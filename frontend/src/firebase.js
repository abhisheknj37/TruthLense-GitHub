import { initializeApp, getApps } from "firebase/app";
import {
  createUserWithEmailAndPassword,
  getAuth,
  GoogleAuthProvider,
  onAuthStateChanged,
  sendEmailVerification,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  updateProfile
} from "firebase/auth";

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET
};

export const firebaseConfigured = Boolean(
  config.apiKey && config.authDomain && config.projectId && config.appId
);

const app = firebaseConfigured
  ? (getApps().length ? getApps()[0] : initializeApp(config))
  : null;

export const auth = app ? getAuth(app) : null;
const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: "select_account" });

export function observeAuth(callback) {
  if (!auth) {
    callback(null);
    return () => {};
  }
  return onAuthStateChanged(auth, callback);
}

function requireAuth() {
  if (!auth) {
    throw new Error("Firebase is not configured. Add the VITE_FIREBASE_* values to frontend/.env.local and restart Vite.");
  }
  return auth;
}

export async function signInEmail(email, password) {
  return signInWithEmailAndPassword(requireAuth(), email.trim(), password);
}

export async function createEmailAccount(name, email, password) {
  const credential = await createUserWithEmailAndPassword(requireAuth(), email.trim(), password);
  if (name.trim()) await updateProfile(credential.user, { displayName: name.trim() });
  try { await sendEmailVerification(credential.user); } catch { /* Verification email may be configured later. */ }
  return credential;
}

export async function signInGoogle() {
  return signInWithPopup(requireAuth(), googleProvider);
}

export async function signOutUser() {
  return signOut(requireAuth());
}
