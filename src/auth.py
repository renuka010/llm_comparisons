import bcrypt
import streamlit as st
from src.config import cfg
from src.db import get_db, init_db


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def ensure_demo_user() -> None:
    if not cfg.APP_USERNAME or not cfg.APP_PASSWORD:
        return
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (cfg.APP_USERNAME,)).fetchone()
        if not row:
            hashed = _hash_password(cfg.APP_PASSWORD)
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (cfg.APP_USERNAME, hashed),
            )


def login(username: str, password: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        return False
    return _verify_password(password, row["password_hash"])


def require_auth() -> bool:
    """Returns True if user is authenticated. Shows login form otherwise."""
    if st.session_state.get("authenticated"):
        return True

    ensure_demo_user()

    if not cfg.APP_USERNAME or not cfg.APP_PASSWORD:
        st.error("APP_USERNAME and APP_PASSWORD secrets are not set. Configure them in Space Settings → Variables and Secrets.")
        return False

    st.markdown("## AI Assistant Comparison")
    st.markdown("Sign in to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if login(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid credentials.")

    return False
