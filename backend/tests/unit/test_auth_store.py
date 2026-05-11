from __future__ import annotations

import auth_store


def _init_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(auth_store, "DB_PATH", tmp_path / "elektroscan-test.db")
    auth_store.init_database()


def test_create_and_authenticate_user(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)

    user = auth_store.create_user(
        email="Test@Example.com",
        password="bezpieczne-haslo",
        name="Tester",
    )

    assert user["email"] == "test@example.com"
    assert user["name"] == "Tester"
    assert auth_store.authenticate_user("test@example.com", "bezpieczne-haslo")["id"] == user["id"]
    assert auth_store.authenticate_user("test@example.com", "zle-haslo") is None


def test_session_token_returns_public_user(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")

    token, session = auth_store.create_auth_session(user["id"])
    current = auth_store.get_user_by_session_token(token)

    assert session["userId"] == user["id"]
    assert current["id"] == user["id"]
    assert "password_hash" not in current


def test_user_profile_update(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")

    updated = auth_store.update_user_profile(user["id"], name="Nowa Nazwa")

    assert updated["name"] == "Nowa Nazwa"


def test_password_reset_changes_password_and_clears_sessions(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")
    session_token, _session = auth_store.create_auth_session(user["id"])

    reset = auth_store.create_password_reset_token_for_email("a@example.com")
    assert reset is not None
    reset_token, reset_info = reset
    changed = auth_store.reset_password_with_token(reset_token, "nowe-haslo-123")

    assert reset_info["purpose"] == "password_reset"
    assert changed["id"] == user["id"]
    assert auth_store.authenticate_user("a@example.com", "password123") is None
    assert auth_store.authenticate_user("a@example.com", "nowe-haslo-123")["id"] == user["id"]
    assert auth_store.get_user_by_session_token(session_token) is None
    assert auth_store.reset_password_with_token(reset_token, "kolejne-haslo-123") is None


def test_auth_sessions_can_be_listed_and_deleted(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")

    current_token, current_session = auth_store.create_auth_session(user["id"])
    other_token, other_session = auth_store.create_auth_session(user["id"])
    sessions = auth_store.list_auth_sessions_for_user(user["id"], current_token=current_token)

    assert {session["id"] for session in sessions} == {current_session["id"], other_session["id"]}
    assert next(session for session in sessions if session["id"] == current_session["id"])["isCurrent"]
    assert auth_store.delete_auth_session_by_id(user["id"], other_session["id"])
    assert auth_store.get_user_by_session_token(other_token) is None
    assert auth_store.delete_auth_sessions_for_user(user["id"], except_token=current_token) == 0
    assert auth_store.get_user_by_session_token(current_token)["id"] == user["id"]


def test_projects_are_scoped_to_owner(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user_a = auth_store.create_user(email="a@example.com", password="password123")
    user_b = auth_store.create_user(email="b@example.com", password="password123")

    project = auth_store.create_project(user_a["id"], name="Projekt A")

    assert auth_store.get_project_for_user(project["id"], user_a["id"]) is not None
    assert auth_store.get_project_for_user(project["id"], user_b["id"]) is None
    assert auth_store.list_projects_for_user(user_b["id"]) == []


def test_project_upload_session_scope(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")
    project = auth_store.create_project(user["id"], name="Projekt A")

    auth_store.record_project_upload_session(
        session_id="session-1",
        project_id=project["id"],
        source_pdf="plan.pdf",
    )
    updated_project = auth_store.get_project_for_user(project["id"], user["id"])

    assert auth_store.project_session_exists(project["id"], "session-1")
    assert not auth_store.project_session_exists(project["id"], "session-x")
    assert updated_project["latestSessionId"] == "session-1"
    assert updated_project["latestSourcePdf"] == "plan.pdf"


def test_analysis_runs_are_recorded_for_project(monkeypatch, tmp_path):
    _init_tmp_db(monkeypatch, tmp_path)
    user = auth_store.create_user(email="a@example.com", password="password123")
    project = auth_store.create_project(user["id"], name="Projekt A")
    auth_store.record_project_upload_session(
        session_id="session-1",
        project_id=project["id"],
        source_pdf="plan.pdf",
    )

    auth_store.record_analysis_run(
        analysis_id="analysis-1",
        project_id=project["id"],
        session_id="session-1",
        source_pdf="plan.pdf",
        snapshot_path="/tmp/analysis-1.json",
    )

    runs = auth_store.list_analysis_runs_for_project(project["id"])
    updated_project = auth_store.get_project_for_user(project["id"], user["id"])

    assert runs[0]["id"] == "analysis-1"
    assert runs[0]["hasSnapshot"]
    assert auth_store.get_analysis_run_for_project(project["id"], "analysis-1")["sourcePdf"] == "plan.pdf"
    assert updated_project["analysisCount"] == 1
    assert updated_project["latestAnalysisAtUtc"]
