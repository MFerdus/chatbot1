"""Offline startup checks; no OpenAI or Pinecone request is made."""
from app import app


def main() -> None:
    client = app.test_client()
    home = client.get("/")
    health = client.get("/health")
    assert home.status_code == 200, home.status
    assert health.status_code in {200, 503}, health.status
    assert client.post("/get", data={"msg": ""}).status_code == 400
    print("Imports, template rendering, health route, and input validation passed.")


if __name__ == "__main__":
    main()
