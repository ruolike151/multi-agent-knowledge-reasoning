"""从巨潮资讯网下载 M0 定死的 15 份语料（10 份年报 + 5 份公告）。

公司名单与筛选规则见设计文档 §8.1 / §17.1 Q-06：光伏产业链 5 家 × 2024/2025 年报，
外加每家公司各 1 份公告，类型覆盖「重大合同 / 对外投资 / 关联交易 / 业绩预告 /
投资者关系活动记录」。

用法（解释器为仓库内 .venv）：
    python scripts/fetch_corpus.py --dry-run     # 只打印选中的条目，不下载
    python scripts/fetch_corpus.py               # 下载 PDF 并写 data/manifest.json

产出：
    data/corpus/annual/<code>_<公司名>_<年>年报.pdf        （10 份）
    data/corpus/announcement/<code>_<公司名>_<类型>.pdf    （5 份）
    data/manifest.json                                     （doc_id / 公司名 / 报告期 / 页数 ...）

选取流程（可复现，全部以巨潮公告列表的元数据为准）：
     1. 用 topSearch 接口按股票代码换 orgId；
     2. 年报：按 `category_ndbg_szsh` 拉全量，排除「摘要 / 英文 / 取消」，取中文全文本；
     3. 公告：拉 2025-01-01~2026-12-31 全量公告，按类型关键词 + 噪声排除归入 5 个桶；
     4. 5 种类型用回溯匹配指派给 5 家不同公司（候选少的类型先分配）；
     5. 每个槽位按标题时间倒序逐条试下载（最多 4 条），按 `score_announcement`
        打分（3 分 = 含金额/产能 且 点名本批其他公司；2 分 = 含金额/产能 且有公司实体），
        留下得分最高的一份，其余候选文件删除。

接口注意：查询接口固定每页返回 30 条且忽略 pageSize（第一页的 totalAnnouncement
可能为 0，故取累计最大值翻页）；`static.cninfo.com.cn` 的 HTTPS 在部分 Windows
环境下 schannel 握手失败，因此 PDF 一律走 requests 下载。

合规：年报与公告均为公开披露文件，仅个人学习研究使用；PDF 不入 Git（`data/` 已被忽略），
只提交本脚本与 `data/manifest.json`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
import requests

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
TOPSEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
STATIC_BASE = "https://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
}

REQUEST_PAUSE_SEC = 0.5
ANNUAL_YEARS = (2024, 2025)
SE_DATE = "2025-01-01~2026-12-31"


@dataclass(frozen=True)
class Company:
    code: str
    name: str
    chain: str

    @property
    def column(self) -> str:
        """巨潮的板块参数：6 开头为沪市，0/3 开头为深市。"""
        return "sse" if self.code.startswith("6") else "szse"


# 与设计文档 §8.1 的公司名单逐字对应
COMPANIES = (
    Company("600438", "通威股份", "上游 · 硅料 / 电池片"),
    Company("002129", "TCL中环", "中游 · 硅片"),
    Company("601012", "隆基绿能", "中游 · 硅片 + 组件"),
    Company("002459", "晶澳科技", "中游 · 组件"),
    Company("300274", "阳光电源", "下游 · 逆变器 / 储能"),
)

# 制度、第三方意见、报告类文件不是事件型事实，统一排除
NOISE_RE = re.compile(
    r"管理制度|管理办法|决策制度|议事规则|公司章程|独立董事|法律意见"
    r"|保荐|持续督导|专项报告|专项说明|审计报告|核查意见|关注公告"
    r"|内部控制|会计师事务所|信用评估|英文|摘要"
)

# 公告类型 → (doc_id 后缀, 标题必须匹配, 标题必须排除, 优先关键词)
# 关键词按 2026-09-24 五家公司的真实标题校准（见 docs 中语料下载记录）。
ANNOUNCEMENT_TYPES = (
    (
        "重大合同",
        "contract",
        re.compile(r"重大合同|框架协议|中标|销售协议|采购协议|长单|订单"),
        re.compile(r"取消|终止|解除|补充|募集资金|监管协议|员工持股"),
        re.compile(r"销售|采购|中标|长单|订单"),
    ),
    (
        "对外投资",
        "investment",
        re.compile(r"对外投资|增资|扩产|投资建设|投资项目|设立.*公司"),
        re.compile(r"现金管理|委托理财|风险|减值|意向|终止"),
        re.compile(r"增资|投资建设|扩产|投资项目|设立"),
    ),
    (
        "关联交易",
        "related",
        re.compile(r"关联交易|关联方"),
        re.compile(r"取消|终止|占用"),
        re.compile(r"日常关联交易"),
    ),
    (
        "业绩预告",
        "forecast",
        re.compile(r"业绩预告|业绩预增|业绩预减|业绩预亏|业绩快报"),
        re.compile(r"取消|修正|关注"),
        re.compile(r"\d{4}\s*年度业绩预告"),
    ),
    (
        "投资者关系活动记录",
        "ir",
        re.compile(r"投资者关系|调研|接待|说明会|路演"),
        re.compile(r"制度|办法"),
        re.compile(r"记录表|调研|接待"),
    ),
)

AMOUNT_RE = re.compile(r"\d[\d,\.]*\s*(?:亿元|万元|亿|万|元|GW|MW|万吨|万台)")
# 「提及产业链上下游实体」的可检查代理：正文里出现的公司全称
ENTITY_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z]{2,20}?(?:股份)?有限公司")


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def org_id_of(session: requests.Session, code: str) -> str:
    resp = session.post(
        TOPSEARCH_URL, data={"keyWord": code, "maxNum": 10}, timeout=30
    )
    resp.raise_for_status()
    hits = resp.json()
    for hit in hits:
        if hit.get("code") == code:
            return hit["orgId"]
    raise RuntimeError(f"巨潮未找到股票代码 {code}")


def query_announcements(
    session: requests.Session,
    *,
    code: str,
    org_id: str,
    column: str,
    category: str,
    se_date: str = SE_DATE,
    page_size: int = 30,
    max_pages: int = 40,
) -> list[dict]:
    """按条件分页拉取公告列表。

    注意：巨潮接口忽略 pageSize，固定每页返回 30 条；页数用返回的
    totalAnnouncement 反推，避免只拿到第一页。
    """
    collected: list[dict] = []
    total = 0
    for page in range(1, max_pages + 1):
        resp = session.post(
            QUERY_URL,
            data={
                "pageNum": page,
                "pageSize": page_size,
                "column": column,
                "tabName": "fulltext",
                "plate": "",
                "stock": f"{code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": category,
                "trade": "",
                "seDate": se_date,
                "sortName": "time",
                "sortType": "desc",
                "isHLtitle": "true",
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("announcements") or []
        # 实测第一页的 totalAnnouncement 可能为 0，后续页才给出真实总数，
        # 因此取累计最大值，而不是只在第一页赋值。
        total = max(total, payload.get("totalAnnouncement") or 0)
        collected.extend(batch)
        if not batch or len(collected) >= total:
            break
        time.sleep(REQUEST_PAUSE_SEC)
    return collected


def announcement_date(item: dict) -> str:
    ts = item.get("announcementTime")
    if not ts:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def period_label(item: dict) -> str:
    """按标题推断报告期。

    标题里自带年份时以其为准（业绩预告/日常关联交易预计的年份是**预测期**，
    与披露年份不是一回事，例如 2026-01 披露的「2025年度业绩预告」）。
    """
    title = item.get("clean_title") or item.get("announcementTitle") or ""
    hit = re.search(r"(\d{4})\s*年", title)
    year = hit.group(1) if hit else announcement_date(item)[:4]
    if "半年度" in title or "半年报" in title:
        return f"{year}年半年度"
    quarter = re.search(r"第\s*[一二三四]\s*季度", title)
    if quarter:
        return f"{year}年{quarter.group(0).replace(' ', '')}"
    return f"{year}年度"


def pick_annual_reports(items: list[dict], company: Company) -> dict[int, dict]:
    """从年报列表里挑出中文全文版（排除摘要、英文版、修订版）。"""
    chosen: dict[int, dict] = {}
    for item in items:
        title = (item.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")
        if "英文" in title or "摘要" in title or "取消" in title:
            continue
        match = re.search(r"(\d{4})\s*年年度报告$", title)
        if not match:
            continue
        year = int(match.group(1))
        if year not in ANNUAL_YEARS:
            continue
        chosen.setdefault(year, {**item, "clean_title": title})
    return chosen


def candidate_announcements(items: list[dict]) -> dict[str, list[dict]]:
    """按类型归集候选公告；同类内优先关键词命中者在前，其余按披露日期倒序。"""
    buckets: dict[str, list[dict]] = {
        name: [] for name, _, _, _, _ in ANNOUNCEMENT_TYPES
    }
    for item in items:
        title = (item.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")
        if NOISE_RE.search(title):
            continue
        for name, _slug, include, exclude, _prefer in ANNOUNCEMENT_TYPES:
            if include.search(title) and not exclude.search(title):
                buckets[name].append({**item, "clean_title": title})
    for name, _slug, _include, _exclude, prefer in ANNOUNCEMENT_TYPES:
        buckets[name].sort(
            key=lambda it: (
                bool(prefer.search(it["clean_title"])),
                it.get("announcementTime") or 0,
            ),
            reverse=True,
        )
    return buckets


def assign_types(candidates: dict[str, dict[str, list[dict]]]) -> dict[str, Company]:
    """把 5 种公告类型分别指派给 5 家不同公司（回溯匹配，同类有候选的公司优先）。"""
    names = [name for name, _, _, _, _ in ANNOUNCEMENT_TYPES]
    order = sorted(
        names,
        key=lambda n: min(len(candidates[c.code][n]) or 10**6 for c in COMPANIES),
    )
    assignment: dict[str, Company] = {}
    used: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True
        name = order[index]
        for company in COMPANIES:
            if company.code in used or not candidates[company.code][name]:
                continue
            assignment[name] = company
            used.add(company.code)
            if backtrack(index + 1):
                return True
            used.discard(company.code)
            del assignment[name]
        return False

    if not backtrack(0):
        raise RuntimeError("无法为 5 种公告类型找到 5 家不同公司的候选，请放宽关键词")
    return assignment


def count_pages(path: Path) -> int:
    doc = pdfium.PdfDocument(str(path))
    try:
        return len(doc)
    finally:
        doc.close()


def sample_texts(path: Path, max_pages: int = 2) -> list[str]:
    with pdfplumber.open(path) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages[:max_pages]]


def sample_text(path: Path, max_pages: int = 2) -> str:
    return "\n".join(sample_texts(path, max_pages))


def download(session: requests.Session, item: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = STATIC_BASE + item["adjunctUrl"]
    resp = session.get(url, timeout=180)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"{url} 返回的不是 PDF（前 8 字节：{resp.content[:8]!r}）")
    dest.write_bytes(resp.content)
    time.sleep(REQUEST_PAUSE_SEC)


def score_announcement(doc_type: str, body: str, peers: list[str]) -> tuple[int, dict]:
    """按设计文档 §8.1 的筛选规则给候选公告打分（分值越高越好）。

    规则要求「含明确金额或产能数字、且提及产业链上下游实体」，落到可检查的判据上：
    3 分 = 含金额/产能 且 点名了本批其他公司；2 分 = 含金额/产能 且有公司实体；
    1 分 = 只满足其一；0 分 = 都不满足。

    例外：投资者关系类公告本身很短，五家公司在 2024–2026 窗口内**都没有**发布正式的
    《投资者关系活动记录表》，因此该类放宽为「有公司实体」即可（记 `filter_relaxed`）。
    """
    has_amount = bool(AMOUNT_RE.search(body))
    mentions = [n for n in peers if n in body]
    entities: list[str] = []
    for name in ENTITY_RE.findall(body):
        if name not in entities:
            entities.append(name)
    relaxed = doc_type == "投资者关系活动记录" and not has_amount
    if has_amount and mentions:
        score = 3
    elif has_amount and entities:
        score = 2
    elif has_amount or (relaxed and entities):
        score = 1
    else:
        score = 0
    return score, {
        "has_amount": has_amount,
        "mentions_peers": mentions,
        "entity_names": entities[:10],
        "filter_relaxed": relaxed,
        "filter_score": score,
    }


def select_announcement(
    session: requests.Session,
    candidates: list[dict],
    dest: Path,
    peers: list[str],
    doc_type: str,
    limit: int = 4,
) -> dict | None:
    """逐条试下载候选，留下得分最高的一条（未通过的候选文件会被删除）。

    拿到满分（3 分：金额 + 点名同业）即停止探测；探测完 limit 条仍未满分时，取其中
    最高分者。若候选全为 0 分则返回 None。
    """
    best: dict | None = None
    best_score = 0
    on_disk: dict | None = None
    for item in candidates[:limit]:
        if dest.exists():
            dest.unlink()
        print(f"[候选] {dest.stem} ← {item['clean_title']}")
        download(session, item, dest)
        on_disk = item
        score, info = score_announcement(doc_type, sample_text(dest, max_pages=3), peers)
        if score > best_score:
            best_score = score
            best = {**item, **info}
        if score == 3:
            break
    if best is None or best_score == 0:
        if dest.exists():
            dest.unlink()
        return None
    if on_disk is None or on_disk.get("adjunctUrl") != best.get("adjunctUrl"):
        dest.unlink(missing_ok=True)
        download(session, best, dest)
    return best


def doc_entry(
    *,
    doc_id: str,
    kind: str,
    company: Company,
    period: str,
    item: dict,
    path: Path,
    extra: dict | None = None,
) -> dict:
    pages = count_pages(path)
    # 年报封面页文字很少，只取第 1 页会把正常 PDF 误判成扫描件，因此抽样 3 页。
    head_texts = sample_texts(path, max_pages=3)
    sample_chars = len("\n".join(head_texts).strip())
    entry = {
        "doc_id": doc_id,
        "kind": kind,
        "company": company.name,
        "code": company.code,
        "chain": company.chain,
        "period": period,
        "pages": pages,
        "file": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "announcement_title": item.get("clean_title") or item.get("announcementTitle"),
        "announcement_date": announcement_date(item),
        "source_url": STATIC_BASE + item["adjunctUrl"],
        "text_layer": sample_chars > 200,
        "page1_chars": len(head_texts[0].strip()) if head_texts else 0,
        "sample_chars": sample_chars,
    }
    if extra:
        entry.update(extra)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 M0 语料（巨潮资讯网）")
    parser.add_argument("--out", default="data", help="数据根目录（默认 data）")
    parser.add_argument("--dry-run", action="store_true", help="只打印选择结果，不下载")
    args = parser.parse_args()

    root = Path(args.out)
    session = new_session()
    entries: list[dict] = []
    problems: list[str] = []

    annual_items: dict[str, dict[int, dict]] = {}
    announcement_items: dict[str, dict[str, list[dict]]] = {}

    for company in COMPANIES:
        org_id = org_id_of(session, company.code)
        print(f"[查询] {company.code} {company.name} orgId={org_id}")
        annual = query_announcements(
            session,
            code=company.code,
            org_id=org_id,
            column=company.column,
            category="category_ndbg_szsh",
        )
        picked = pick_annual_reports(annual, company)
        missing = [y for y in ANNUAL_YEARS if y not in picked]
        if missing:
            problems.append(f"{company.name} 缺年报：{missing}")
        annual_items[company.code] = picked

        all_notices = query_announcements(
            session,
            code=company.code,
            org_id=org_id,
            column=company.column,
            category="",
        )
        announcement_items[company.code] = candidate_announcements(all_notices)
        counts = {k: len(v) for k, v in announcement_items[company.code].items()}
        print(f"        年报 {sorted(picked)}；公告候选 {counts}")
        time.sleep(REQUEST_PAUSE_SEC)

    assignment = assign_types(
        {code: buckets for code, buckets in announcement_items.items()}
    )
    print("[匹配] 公告类型指派：" + "，".join(f"{t}→{c.name}" for t, c in assignment.items()))

    if args.dry_run:
        for company in COMPANIES:
            for year in ANNUAL_YEARS:
                item = annual_items[company.code].get(year)
                if item:
                    print(f"  年报 {company.code} {year}：{item['clean_title']} @ {announcement_date(item)}")
        for name, company in assignment.items():
            item = announcement_items[company.code][name][0]
            print(f"  公告 {company.code} {name}：{item['clean_title']} @ {announcement_date(item)}")
        return 0

    for company in COMPANIES:
        for year in ANNUAL_YEARS:
            item = annual_items[company.code].get(year)
            if not item:
                continue
            dest = root / "corpus" / "annual" / f"{company.code}_{company.name}_{year}年报.pdf"
            print(f"[下载] {dest.name}")
            download(session, item, dest)
            entries.append(
                doc_entry(
                    doc_id=f"annual-{company.code}-{year}",
                    kind="annual",
                    company=company,
                    period=f"{year}年度",
                    item=item,
                    path=dest,
                )
            )

    for name, company in assignment.items():
        candidates = announcement_items[company.code][name]
        slug = next(s for t, s, _, _, _ in ANNOUNCEMENT_TYPES if t == name)
        others = [c.name for c in COMPANIES if c.code != company.code]
        dest = root / "corpus" / "announcement" / f"{company.code}_{company.name}_{name}.pdf"
        chosen = select_announcement(session, candidates, dest, others, name)
        if chosen is None:
            problems.append(f"{company.name} 的「{name}」没有通过筛选规则的候选")
            continue
        print(f"[下载] {dest.name} ← {chosen['clean_title']} @ {announcement_date(chosen)}")
        entries.append(
            doc_entry(
                doc_id=f"ann-{company.code}-{slug}",
                kind="announcement",
                company=company,
                period=period_label(chosen),
                item=chosen,
                path=dest,
                extra={
                    "announcement_type": name,
                    "has_amount": chosen["has_amount"],
                    "mentions_peers": chosen["mentions_peers"],
                    "entity_names": chosen["entity_names"],
                    "filter_relaxed": chosen["filter_relaxed"],
                    "filter_score": chosen["filter_score"],
                },
            )
        )

    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": "巨潮资讯网 www.cninfo.com.cn",
        "selection_rule": "设计文档 §8.1 / §17.1 Q-06：光伏产业链 5 家 × 2024/2025 年报 + 每家 1 份公告",
        "companies": [
            {"code": c.code, "name": c.name, "chain": c.chain} for c in COMPANIES
        ],
        "documents": sorted(entries, key=lambda e: (e["kind"], e["code"], e["period"])),
        "problems": problems,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",  # Windows 上写作 LF，保证清单文件跨平台一致
    )

    print(f"\n完成：{len(entries)} 份文档 → {manifest_path.as_posix()}")
    for problem in problems:
        print(f"  [待处理] {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
