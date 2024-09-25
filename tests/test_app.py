import pytest

from ..app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_chat_endpoint(client):
    response = client.post("/chat", json={"query": "Where to sign-up?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data
    assert "confidence" in data


def test_refresh_index_endpoint(client):
    response = client.post("/refresh_index")
    assert response.status_code == 200
    data = response.get_json()
    assert "message" in data


def test_index_status_endpoint(client):
    response = client.get("/index_status")
    assert response.status_code == 200
    data = response.get_json()
    assert "index_exists" in data
    assert "index_path" in data
