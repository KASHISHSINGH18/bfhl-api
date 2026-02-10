import pytest
from httpx import AsyncClient
from main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["is_success"] is True
        assert "official_email" in data


@pytest.mark.asyncio
async def test_fibonacci():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/bfhl", json={"fibonacci": 7})
        assert r.status_code == 200
        data = r.json()
        assert data["is_success"] is True
        assert data["data"] == [0,1,1,2,3,5,8]


@pytest.mark.asyncio
async def test_prime():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/bfhl", json={"prime": [2,4,7,9,11]})
        assert r.status_code == 200
        data = r.json()
        assert data["data"] == [2,7,11]


@pytest.mark.asyncio
async def test_lcm_hcf():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r1 = await ac.post("/bfhl", json={"lcm": [12,18,24]})
        assert r1.status_code == 200
        assert r1.json()["data"] == 72

        r2 = await ac.post("/bfhl", json={"hcf": [24,36,60]})
        assert r2.status_code == 200
        assert r2.json()["data"] == 12


@pytest.mark.asyncio
async def test_invalid_multiple_keys():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/bfhl", json={"fibonacci": 3, "prime": [2]})
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_invalid_payload():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/bfhl", json={"fibonacci": -1})
        assert r.status_code == 400
