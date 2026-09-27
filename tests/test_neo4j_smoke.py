import os

import pytest
from neo4j import GraphDatabase


@pytest.mark.integration
def test_neo4j_is_reachable():
    # 兜底值用 127.0.0.1：写 localhost 会先试 ::1:7687（WSL2 端口转发接管，不回 Bolt 握手）
    # 而卡满 60s 超时，见内部偏差日志第二十三节。
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "akr-local-pass"))
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session() as session:
            assert session.run("RETURN 1 AS ok").single()["ok"] == 1
    finally:
        driver.close()
