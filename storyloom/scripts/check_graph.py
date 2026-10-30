# /// script
# requires-python = ">=3.10"
# ///
"""检查“故事图谱”有没有逻辑漏洞（Storyloom 的检查脚本）。

用法：
  uv run check_graph.py <剧的文件夹>   读 <剧的文件夹>/故事图谱.json；
                                        有 EP01/剧本.md 这样的文件也读，用来核对段号
  uv run check_graph.py --selftest      用脚本里自带的小样本自测（不联网、不调用模型）

输出分三级：✗ 必改 / ⚠ 提醒 / ✓ 小计。有“必改”时退出码是 1，否则是 0。
字段含义见 ../references/故事图谱.md。只用标准库。
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

MUST, WARN = "必改", "提醒"
OPEN_MINUTES = 15  # 没写 until 的事件，最多算 15 分钟

# 时间一律是“分钟数”。没写秒就是整数；写了秒（D2 02:00:43）就带小数，用分数存，比较、相加都不会有浮点误差。
Minutes = int | Fraction
TIME_RE = re.compile(r"^D(\d+)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$")
# 下面三个跟 blind_pack.py 保持一致：段标题前后可能带【新增】【改】【删】，只在“## 正文”那一节里找
SEG_RE = re.compile(r"^(?:【(?:新增|改|删)】\s*)?\*\*第\s*(\d+)\s*段.*$", re.M)
BODY_RE = re.compile(r"^##\s*正文\s*$", re.M)
NEXT_H2_RE = re.compile(r"^##\s+\S", re.M)

Pos = tuple[int, "int | None"]  # (集, 段)；段没写就是 None

# 问题的种类：代码 -> (级别, 名字)。顺序就是输出时的排列顺序，也是小计里的顺序。
CODES: dict[str, tuple[str, str]] = {
    "bad_type": (MUST, "字段类型不对"),
    "missing_field": (MUST, "必填字段缺了"),
    "dup_id": (MUST, "id 重复"),
    "bad_ref": (MUST, "引用了不存在的 id"),
    "bad_time": (MUST, "时间格式不对"),
    "until_before_at": (MUST, "until 早于 at"),
    "two_places": (MUST, "一人两地"),
    "learns_no_from": (MUST, "知道没有来源"),
    "perceive_far": (MUST, "看见听见说不通"),
    "perceive_unsure": (WARN, "看见听见没法核对"),
    "walk_short": (WARN, "来不及走过去"),
    "walk_unreachable": (WARN, "路不通"),
    "payoff_no_planted": (MUST, "用上了没埋下"),
    "payoff_before_planted": (MUST, "用上早于埋下"),
    "clue_no_arc": (WARN, "线索没写 arc"),
    "early_exposure": (WARN, "可能太早暴露凶手"),
    "early_exposure_light": (WARN, "可能太早暴露凶手（较轻）"),
    "unused_clue": (WARN, "埋了没用上"),
    "herring_open": (WARN, "误导没推翻"),
    "debunk_early": (WARN, "推翻早于埋下"),
    "single_proof": (WARN, "孤证"),
    "no_proof": (WARN, "没有线索指向凶手"),
    "spoiler_must": (MUST, "没上屏却没标 spoiler（牵涉凶手）"),
    "spoiler_warn": (WARN, "没上屏却没标 spoiler"),
    "name_leak": (WARN, "文字里有凶手名字"),
    "seg_missing": (MUST, "段号在剧本里找不到"),
    "no_script": (WARN, "没有剧本文件"),
    "script_unparsed": (WARN, "剧本认不出段标题"),
}


@dataclass
class Issue:
    code: str
    text: str
    arc: str | None = None  # 属于哪条故事线（小计里按故事线分开数）

    @property
    def level(self) -> str:
        return CODES[self.code][0]

    @property
    def label(self) -> str:
        return CODES[self.code][1]


class Report:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, code: str, text: str, arc: str | None = None) -> None:
        assert code in CODES, code
        self.issues.append(Issue(code, text, arc if isinstance(arc, str) else None))

    def by_level(self, level: str) -> list[Issue]:
        order = {c: i for i, c in enumerate(CODES)}
        return sorted((x for x in self.issues if x.level == level), key=lambda x: order[x.code])

    def codes(self) -> set[str]:
        return {x.code for x in self.issues}

    def count(self, code: str) -> int:
        return sum(1 for x in self.issues if x.code == code)


# ---------------------------------------------------------------- 小工具

def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def blank(v: Any) -> bool:
    """没写、写成空的，都算缺。数字（包括 0）和 False 不算缺。"""
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and not v)


def sid(v: Any) -> str | None:
    """当 id 用的值：必须是非空文字。"""
    return v if isinstance(v, str) and v.strip() else None


def tname(v: Any) -> str:
    names = {str: "文字", int: "整数", float: "小数", bool: "是/否", list: "列表", dict: "对象", type(None): "空"}
    return names.get(type(v), type(v).__name__)


def parse_time(v: Any) -> Minutes | None:
    """“D2 02:05”（可以带秒：“D2 02:05:30”）-> 从第 0 天零点起的分钟数；格式不对返回 None。"""
    if not isinstance(v, str):
        return None
    m = TIME_RE.match(v.strip())
    if not m:
        return None
    day, hh, mm, ss = int(m[1]), int(m[2]), int(m[3]), int(m[4] or 0)
    if hh > 23 or mm > 59 or ss > 59:
        return None
    t = day * 1440 + hh * 60 + mm
    return t + Fraction(ss, 60) if ss else t


def fmt_time(t: Minutes) -> str:
    """有秒才显示秒：D2 02:05 / D2 02:05:30。"""
    d, r = divmod(round(t * 60), 86400)
    hm = f"D{d} {r // 3600:02d}:{r % 3600 // 60:02d}"
    return f"{hm}:{r % 60:02d}" if r % 60 else hm


def fmt_span(a: Minutes, b: Minutes) -> str:
    end = fmt_time(b)
    return f"{fmt_time(a)}–{end.split(' ')[1] if a // 1440 == b // 1440 else end}"


def fmt_gap(m: Minutes) -> str:
    """一段时间：整分钟写“3 分钟”，带秒写“3 分 20 秒”。"""
    s = round(m * 60)
    if s % 60 == 0:
        return f"{s // 60} 分钟"
    return f"{s // 60} 分 {s % 60} 秒" if s >= 60 else f"{s} 秒"


def snippet(s: Any, n: int = 14) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def fmt_pos(p: Pos) -> str:
    ep, seg = p
    return f"第 {ep} 集第 {seg} 段" if seg is not None else f"第 {ep} 集"


def pos_cmp(a: Pos, b: Pos) -> int | None:
    """按（集、段）比先后：-1 在前，0 同一处，1 在后；比不出来（同一集但有一边没写段）返回 None。"""
    if a[0] != b[0]:
        return -1 if a[0] < b[0] else 1
    if a[1] is None or b[1] is None:
        return None
    return (a[1] > b[1]) - (a[1] < b[1])


def pos_key(p: Pos) -> tuple[int, int]:
    return (p[0], p[1] if p[1] is not None else 0)


def fmt_ranges(nums: Any) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}–{nums[j]}")
        i = j + 1
    return "、".join(out)


def short_arc_name(name: str) -> str:
    """“案子1 · 明早九点见（草稿）” -> “案子1”"""
    return name.split("·")[0].strip() or name


# ---------------------------------------------------------------- 读进来、建索引、做“基础”检查

KINDS = {
    "characters": "人物", "relations": "关系", "maps": "地图", "locations": "地点", "links": "links",
    "arcs": "故事线", "episodes": "集", "clues": "线索", "events": "事件",
}
REQUIRED = {  # 格式说明里标了“必填”的字段
    "characters": ["id", "name"],
    "relations": ["from", "to", "label"],
    "locations": ["id", "name"],
    "arcs": ["id", "name"],
    "episodes": ["n"],
    "clues": ["id", "name"],
    "events": ["id", "at", "who", "where", "what"],
}
HINT = {
    "id": "编号", "name": "名字", "from": "起点人物", "to": "终点人物", "label": "关系说明",
    "n": "第几集", "at": "发生的时间", "who": "涉及的人", "where": "发生的地点", "what": "在干什么",
    "arc": "属于哪条故事线", "text": "文字", "version": "版本号", "title": "剧名",
}


def make_label(key: str, i: int, it: dict) -> str:
    kind = KINDS[key]
    if key == "episodes":
        n = it.get("n")
        return f"第 {n} 集" if is_int(n) else f"第 {i + 1} 个集"
    if key in ("relations", "links"):
        a, b = sid(it.get("from")), sid(it.get("to"))
        if a and b:
            return f"关系 {a}→{b}" if key == "relations" else f"links {a}–{b}"
        return f"第 {i + 1} 条{kind}"
    v = sid(it.get("id"))
    if not v:
        return f"第 {i + 1} 个{kind}（没有 id）"
    if key == "clues" and sid(it.get("name")):
        return f"线索 {v}「{snippet(it['name'], 16)}」"
    return f"{kind} {v}"


class Net:
    """地点之间的连接（links）。连接不分方向。"""

    def __init__(self, links: list[dict]) -> None:
        self.pairs: dict[frozenset[str], list[dict]] = {}
        self.adj: dict[str, dict[str, float]] = {}
        for l in links:
            a, b = sid(l.get("from")), sid(l.get("to"))
            if not a or not b or a == b:
                continue
            self.pairs.setdefault(frozenset((a, b)), []).append(l)
            w = l.get("walk")
            w = w if isinstance(w, (int, float)) and not isinstance(w, bool) and w >= 0 else 1  # 不写当 1
            for x, y in ((a, b), (b, a)):
                cur = self.adj.setdefault(x, {})
                if y not in cur or w < cur[y]:
                    cur[y] = w

    def direct(self, a: str, b: str) -> list[dict]:
        return self.pairs.get(frozenset((a, b)), [])

    def shortest(self, a: str, b: str) -> tuple[float, list[str]] | None:
        """走过去最少几分钟，以及经过哪些地点；连不通返回 None。"""
        if a == b:
            return 0, [a]
        dist: dict[str, float] = {a: 0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if u == b:
                break
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        if b not in dist:
            return None
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return dist[b], path[::-1]


@dataclass
class Ev:  # 事件
    item: dict
    id: str
    label: str
    at: Minutes | None
    until: Minutes | None  # 写了、格式对、而且不早于 at 才算
    where: str | None
    who: list[str]
    pos: Pos | None
    offscreen: bool  # ep、seg 都没写 = 没上屏
    spoiler: bool
    perceives: list[dict]
    learns: list[dict]


@dataclass
class Cl:  # 线索
    item: dict
    id: str
    label: str
    arc: str | None
    planted: Pos | None
    payoff: Pos | None
    debunked: Pos | None
    points: list[str]
    real: bool  # real 不是 false 就算真线索
    spoiler: bool


@dataclass
class Bt:  # 节拍
    item: dict
    label: str
    arc: str | None
    pos: Pos
    kind: str | None
    spoiler: bool
    text: str


class Ctx:
    def __init__(self, data: dict[str, Any], r: Report) -> None:
        self.data, self.r = data, r
        self.coll: dict[str, list[tuple[int, dict, str]]] = {k: self._take(k) for k in KINDS}
        for f in ("version", "title"):
            if blank(data.get(f)):
                r.add("missing_field", f"故事图谱最上面：缺少必填字段 {f}（{HINT[f]}）")
        for key, fields in REQUIRED.items():
            for _i, it, lab in self.coll[key]:
                for f in fields:
                    if blank(it.get(f)):
                        r.add("missing_field", f"{lab}：缺少必填字段 {f}（{HINT.get(f, f)}）")
        self.chars = self._index("characters")
        self.locs = self._index("locations")
        self.arcs = self._index("arcs")
        self.maps = self._index("maps")
        self._index("clues")
        self._index("events")
        self.layers: dict[str, set[str]] = {}
        self.culprits: dict[str, list[str]] = {}
        self.clears: list[tuple[str, str, Pos]] = []  # (说明, 故事线, 位置)：嫌疑人的 clearedAt
        self.net: Net | None = None
        self.eps: set[int] = set()
        self.evs: list[Ev] = []
        self.cls: list[Cl] = []
        self.bts: list[Bt] = []
        self._read_world()
        self._read_story()
        self._read_events()

    # ---- 取列表、建索引
    def _take(self, key: str) -> list[tuple[int, dict, str]]:
        raw = self.data.get(key)
        if raw is None:
            return []
        if not isinstance(raw, list):
            self.r.add("bad_type", f"{key}：应该是一个列表，现在是{tname(raw)}")
            return []
        out = []
        for i, it in enumerate(raw):
            if not isinstance(it, dict):
                self.r.add("bad_type", f"{key} 第 {i + 1} 项：应该是一个对象 {{…}}，现在是{tname(it)}")
                continue
            out.append((i, it, make_label(key, i, it)))
        return out

    def _index(self, key: str) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        first: dict[str, int] = {}
        for i, it, lab in self.coll[key]:
            v = it.get("id")
            if blank(v):
                continue  # 缺 id 已经报过了
            if not isinstance(v, str):
                self.r.add("bad_type", f"{lab}：id 应该是文字，现在是{tname(v)} {v!r}")
                continue
            if v in idx:
                self.r.add("dup_id", f"{lab}：id {v} 重复了（第 {first[v] + 1} 项和第 {i + 1} 项都叫这个）")
            else:
                idx[v], first[v] = it, i
        return idx

    # ---- 好用的显示
    def P(self, pid: str) -> str:
        nm = (self.chars.get(pid) or {}).get("name")
        return f"{pid}({nm})" if isinstance(nm, str) and nm.strip() and nm != pid else str(pid)

    def L(self, lid: str) -> str:
        nm = (self.locs.get(lid) or {}).get("name")
        return f"{lid}({nm})" if isinstance(nm, str) and nm.strip() and nm != lid else str(lid)

    def arc_title(self, aid: str | None) -> str:
        nm = (self.arcs.get(aid) or {}).get("name") if aid else None
        return f"故事线 {aid}（{short_arc_name(nm)}）" if isinstance(nm, str) and nm.strip() else f"故事线 {aid}"

    # ---- 字段读取工具（读的时候顺便报类型错）
    def need_ref(self, label: str, field_: str, value: Any, table: dict, what: str, arc: str | None = None,
                 allow: tuple[str, ...] = ()) -> None:
        if value is None or value == "":
            return
        if not isinstance(value, str):
            self.r.add("bad_type", f"{label}：{field_} 应该是{what} id（文字），现在是{tname(value)} {value!r}", arc)
        elif value not in table and value not in allow:
            self.r.add("bad_ref", f"{label}：{field_} 写的{what} {value} 不存在（{what}表里没有这个 id）", arc)

    def id_list(self, label: str, field_: str, value: Any, arc: str | None = None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            self.r.add("bad_type", f"{label}：{field_} 应该是列表，现在是一段文字 {value!r}（先当只有这一个）", arc)
            return [value] if value.strip() else []
        if not isinstance(value, list):
            self.r.add("bad_type", f"{label}：{field_} 应该是列表，现在是{tname(value)}", arc)
            return []
        out = []
        for x in value:
            if sid(x):
                out.append(x)
            else:
                self.r.add("bad_type", f"{label}：{field_} 里有一项不是 id 文字：{x!r}", arc)
        return list(dict.fromkeys(out))

    def read_pos(self, label: str, field_: str, value: Any, arc: str | None = None) -> Pos | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            self.r.add("bad_type", f"{label}：{field_} 应该写成 {{\"ep\": 1, \"seg\": 7}}，现在是{tname(value)} {value!r}", arc)
            return None
        ep, seg = value.get("ep"), value.get("seg")
        if not is_int(ep):
            why = "没写 ep（第几集）" if ep is None else f"的 ep 应该是整数，现在是 {ep!r}"
            self.r.add("bad_type", f"{label}：{field_} {why}", arc)
            return None
        if seg is not None and not is_int(seg):
            self.r.add("bad_type", f"{label}：{field_} 的 seg 应该是整数，现在是 {seg!r}", arc)
            seg = None
        return (ep, seg)

    def flag(self, label: str, it: dict, name: str, arc: str | None = None) -> bool:
        v = it.get(name)
        if v is not None and not isinstance(v, bool):
            self.r.add("bad_type", f"{label}：{name} 应该是 true 或 false，现在是 {v!r}", arc)
        return v is True

    # ---- 人物、关系、地图、地点、links、故事线
    def _read_world(self) -> None:
        r = self.r
        for _i, it, lab in self.coll["characters"]:
            self.need_ref(lab, "arc", it.get("arc"), self.arcs, "故事线")
        for _i, it, lab in self.coll["relations"]:
            self.need_ref(lab, "from", it.get("from"), self.chars, "人物")
            self.need_ref(lab, "to", it.get("to"), self.chars, "人物")
            self.need_ref(lab, "arc", it.get("arc"), self.arcs, "故事线")
        for _i, it, lab in self.coll["maps"]:
            mid, layers = sid(it.get("id")), it.get("layers")
            if layers is not None and not isinstance(layers, list):
                r.add("bad_type", f"{lab}：layers 应该是列表，现在是{tname(layers)}")
                layers = []
            ids: set[str] = set()
            for l in (layers or []):
                lid = sid(l.get("id")) if isinstance(l, dict) else None
                if lid and lid in ids:
                    r.add("dup_id", f"{lab}：楼层 id {lid} 重复了")
                if lid:
                    ids.add(lid)
            if mid:
                self.layers[mid] = ids
        for _i, it, lab in self.coll["locations"]:
            m, ly = it.get("map"), it.get("layer")
            self.need_ref(lab, "map", m, self.maps, "地图")
            if sid(ly):
                if sid(m) and m in self.maps:
                    if ly not in self.layers.get(m, set()):
                        r.add("bad_ref", f"{lab}：layer 写的楼层 {ly} 不存在（地图 {m} 里没有这一层）")
                elif not sid(m) and not any(ly in s for s in self.layers.values()):
                    r.add("bad_ref", f"{lab}：layer 写的楼层 {ly} 不存在（没有哪张地图有这一层）")
            elif ly is not None and ly != "":
                r.add("bad_type", f"{lab}：layer 应该是楼层 id（文字），现在是{tname(ly)} {ly!r}")
        links = []
        for _i, it, lab in self.coll["links"]:
            self.need_ref(lab, "from", it.get("from"), self.locs, "地点")
            self.need_ref(lab, "to", it.get("to"), self.locs, "地点")
            w = it.get("walk")
            if w is not None and (not isinstance(w, (int, float)) or isinstance(w, bool) or w < 0):
                r.add("bad_type", f"{lab}：walk 应该是分钟数（不小于 0 的数字），现在是 {w!r}（按 1 分钟算）")
            for f in ("see", "hear"):
                if it.get(f) is not None and not isinstance(it.get(f), bool):
                    r.add("bad_type", f"{lab}：{f} 应该是 true 或 false，现在是 {it.get(f)!r}")
            links.append(it)
        if links:
            self.net = Net(links)
        for _i, it, lab in self.coll["arcs"]:
            aid = sid(it.get("id"))
            cul = self.id_list(lab, "culprit", it.get("culprit"), aid)
            for c in cul:
                self.need_ref(lab, "culprit", c, self.chars, "人物", aid)
            if aid:
                self.culprits[aid] = cul
            sus = it.get("suspects")
            if sus is not None and not isinstance(sus, list):
                r.add("bad_type", f"{lab}：suspects 应该是列表，现在是{tname(sus)}", aid)
                sus = []
            for k, s in enumerate(sus or []):
                if not isinstance(s, dict):
                    r.add("bad_type", f"{lab}：suspects 第 {k + 1} 项应该是对象，现在是{tname(s)}", aid)
                    continue
                self.need_ref(lab, f"suspects[{k}].who", s.get("who"), self.chars, "人物", aid)
                pos = self.read_pos(lab, f"suspects[{k}].clearedAt", s.get("clearedAt"), aid)
                if pos and aid:
                    self.clears.append((f"{lab} 的嫌疑人 {s.get('who')} 的 clearedAt（排除嫌疑的位置）", aid, pos))

    # ---- 集、节拍、线索
    def _read_story(self) -> None:
        r = self.r
        seen: set[int] = set()
        for _i, it, lab in self.coll["episodes"]:
            n = it.get("n")
            if n is None:
                continue
            if not is_int(n):
                r.add("bad_type", f"{lab}：n 应该是整数（第几集），现在是{tname(n)} {n!r}")
                continue
            if n in seen:
                r.add("dup_id", f"{lab}：第 {n} 集写了不止一次")
            seen.add(n)
            self.eps.add(n)
            beats = it.get("beats")
            if beats is None:
                continue
            if not isinstance(beats, list):
                r.add("bad_type", f"{lab}：beats 应该是列表，现在是{tname(beats)}")
                continue
            for k, b in enumerate(beats):
                if not isinstance(b, dict):
                    r.add("bad_type", f"{lab}的第 {k + 1} 个节拍：应该是对象，现在是{tname(b)}")
                    continue
                text = b.get("text")
                blab = f"{lab}的节拍 #{k + 1}" + (f"「{snippet(text)}」" if sid(text) else "")
                arc = sid(b.get("arc"))
                for f in ("arc", "text"):
                    if blank(b.get(f)):
                        r.add("missing_field", f"{blab}：缺少必填字段 {f}（{HINT[f]}）", arc)
                self.need_ref(blab, "arc", b.get("arc"), self.arcs, "故事线", arc)
                seg = b.get("seg")
                if seg is not None and not is_int(seg):
                    r.add("bad_type", f"{blab}：seg 应该是整数（第几段），现在是 {seg!r}", arc)
                    seg = None
                kind = b.get("kind") if isinstance(b.get("kind"), str) else None
                self.bts.append(Bt(b, blab, arc, (n, seg), kind, self.flag(blab, b, "spoiler", arc),
                                   text if isinstance(text, str) else ""))
        for _i, it, lab in self.coll["clues"]:
            arc = sid(it.get("arc"))
            self.need_ref(lab, "arc", it.get("arc"), self.arcs, "故事线", arc)
            pts = self.id_list(lab, "pointsTo", it.get("pointsTo"), arc)
            for p in pts:
                self.need_ref(lab, "pointsTo", p, self.chars, "人物", arc)
            real = it.get("real")
            if real is not None and not isinstance(real, bool):
                r.add("bad_type", f"{lab}：real 应该是 true 或 false，现在是 {real!r}", arc)
            self.cls.append(Cl(
                it, sid(it.get("id")) or "", lab, arc,
                self.read_pos(lab, "planted", it.get("planted"), arc),
                self.read_pos(lab, "payoff", it.get("payoff"), arc),
                self.read_pos(lab, "debunked", it.get("debunked"), arc),
                pts, real is not False, self.flag(lab, it, "spoiler", arc)))

    # ---- 事件
    def _dict_list(self, label: str, field_: str, value: Any) -> list[dict]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.r.add("bad_type", f"{label}：{field_} 应该是列表，现在是{tname(value)}")
            return []
        out = []
        for k, x in enumerate(value):
            if isinstance(x, dict):
                out.append(x)
            else:
                self.r.add("bad_type", f"{label}：{field_} 第 {k + 1} 项应该是对象，现在是{tname(x)}")
        return out

    def _read_events(self) -> None:
        r = self.r
        for i, it, lab in self.coll["events"]:
            eid = sid(it.get("id")) or f"#{i + 1}"
            at_raw, until_raw = it.get("at"), it.get("until")
            at, until = parse_time(at_raw), parse_time(until_raw)
            if not blank(at_raw) and at is None:
                r.add("bad_time", f"{lab}：at 写的 {at_raw!r} 不是“D<天> HH:MM”或“D<天> HH:MM:SS”格式（比如 D2 02:00 或 D2 02:00:43）")
            if not blank(until_raw) and until is None:
                r.add("bad_time", f"{lab}：until 写的 {until_raw!r} 不是“D<天> HH:MM”或“D<天> HH:MM:SS”格式（比如 D2 02:05 或 D2 02:05:30）")
            elif until is not None and at is not None and until < at:
                r.add("until_before_at", f"{lab}：until（{fmt_time(until)}）早于 at（{fmt_time(at)}），结束时间跑到开始时间前面了")
                until = None
            where = it.get("where")
            self.need_ref(lab, "where", where, self.locs, "地点")
            who = self.id_list(lab, "who", it.get("who"))
            for p in who:
                self.need_ref(lab, "who", p, self.chars, "人物")
            ep, seg = it.get("ep"), it.get("seg")
            offscreen = ep is None and seg is None
            if ep is not None and not is_int(ep):
                r.add("bad_type", f"{lab}：ep 应该是整数（第几集），现在是 {ep!r}")
                ep = None
            if seg is not None and not is_int(seg):
                r.add("bad_type", f"{lab}：seg 应该是整数（第几段），现在是 {seg!r}")
                seg = None
            per = self._dict_list(lab, "perceives", it.get("perceives"))
            for k, p in enumerate(per):
                self.need_ref(f"{lab} 的 perceives 第 {k + 1} 项", "who", p.get("who"), self.chars, "人物",
                              allow=("audience",))
            lrn = self._dict_list(lab, "learns", it.get("learns"))
            for k, p in enumerate(lrn):
                self.need_ref(f"{lab} 的 learns 第 {k + 1} 项", "who", p.get("who"), self.chars, "人物",
                              allow=("audience",))
            self.evs.append(Ev(it, eid, lab, at, until, sid(where), who,
                               (ep, seg) if ep is not None else None, offscreen,
                               self.flag(lab, it, "spoiler"), per, lrn))


# ---------------------------------------------------------------- 一人两地、知道的来源、看见听见、路走得通

@dataclass
class Stay:  # 一个人的一段时间（来自一个事件）
    pid: str
    ev: Ev
    start: Minutes
    until: Minutes | None
    where: str
    seq: int
    end: Minutes = 0  # 算出来的结束：写了 until 就是 until；没写，到他下一个事件为止，最长 15 分钟


def build_timelines(c: Ctx) -> dict[str, list[Stay]]:
    by: dict[str, list[Stay]] = {}
    for seq, e in enumerate(c.evs):
        if e.at is None or not e.where:
            continue
        for p in e.who:
            by.setdefault(p, []).append(Stay(p, e, e.at, e.until, e.where, seq))
    for lst in by.values():
        # 同一时刻的按列表顺序排：同一分钟里连着写的几件事是个快速的先后，除了最后一件都是零长度
        lst.sort(key=lambda s: (s.start, s.seq))
        for k, s in enumerate(lst):
            if s.until is not None:
                s.end = s.until
            else:
                end = s.start + OPEN_MINUTES
                if k + 1 < len(lst):
                    end = min(end, lst[k + 1].start)
                s.end = max(end, s.start)
    return by


def positions_at(lst: list[Stay], t: Minutes) -> list[Stay]:
    return [s for s in lst if s.start <= t < s.end or s.start == s.end == t]


def check_two_places(c: Ctx, tl: dict[str, list[Stay]]) -> None:
    """同一个人在重叠的时间出现在两个不同地点。首尾相接不算重叠；零长度的事件只在严格落在别人中间时才算。"""
    for pid, lst in tl.items():
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                if b.start >= a.end:
                    break  # 按开始时间排过序，后面的更晚，不会再和 a 重叠
                if a.where == b.where:
                    continue
                if not (b.end > b.start or b.start > a.start):
                    continue
                hi = min(a.end, b.end)
                when = fmt_span(b.start, hi) if hi > b.start else fmt_time(b.start)
                c.r.add("two_places",
                        f"{a.ev.label}：{c.P(pid)} 在 {when} 同时在 {c.L(a.where)} 和 {c.L(b.where)}"
                        f"，一个人不能同时在两个地方（另见{b.ev.label}）")


def check_learns(c: Ctx) -> None:
    for e in c.evs:
        for p in e.learns:
            if blank(p.get("from")):
                who = sid(p.get("who")) or "?"
                who = "观众" if who == "audience" else c.P(who)
                c.r.add("learns_no_from",
                        f"{e.label}：learns 里 {who} 知道了「{snippet(p.get('fact', ''), 20)}」，却没写 from"
                        f"（从哪知道的：亲眼看见、亲耳听见、谁告诉的、看到什么东西）")


def check_perceives(c: Ctx, tl: dict[str, list[Stay]]) -> None:
    for e in c.evs:
        if e.at is None or not e.where or e.where not in c.locs:
            continue
        for p in e.perceives:
            how, who = p.get("how"), sid(p.get("who"))
            if how not in ("看见", "听见") or not who or who == "audience" or who not in c.chars:
                continue
            if who in e.who:
                continue  # 他自己就在这件事里
            need = "see" if how == "看见" else "hear"
            what = snippet(p.get("what", ""), 16)
            head = f"{e.label}：{c.P(who)}「{how}」了「{what}」"
            cands = [s for s in positions_at(tl.get(who, []), e.at) if s.where in c.locs]
            if not cands:
                c.r.add("perceive_unsure",
                        f"{head}，但图谱里不知道 {fmt_time(e.at)} 这一刻 {who} 在哪"
                        f"（没有事件覆盖这一刻；没写 until 的事件最多只算 15 分钟），没法核对")
                continue
            places = list(dict.fromkeys(s.where for s in cands))
            if e.where in places:
                continue
            where_txt = "、".join(c.L(x) for x in places)
            if c.net is None:
                c.r.add("perceive_unsure",
                        f"{head}，当时在 {where_txt}，事件发生在 {c.L(e.where)}；"
                        f"文件里没有 links，只能核对“同一个地点”，这条没法判断")
                continue
            ok, linked = False, False
            for x in places:
                for lk in c.net.direct(x, e.where):
                    linked = True
                    ok = ok or lk.get(need) is True
            if not ok:
                why = (f"links 里它们是连着的，但 {need} 没标 true" if linked
                       else "links 里它们没有直接相连")
                c.r.add("perceive_far",
                        f"{head}，但当时在 {where_txt}，事件发生在 {c.L(e.where)}，{why}，"
                        f"隔着{'看' if how == '看见' else '听'}不到")


def check_walks(c: Ctx, tl: dict[str, list[Stay]]) -> None:
    """同一个人前后两个事件换了地点：按 links 的 walk 算最短路，时间不够走过去就提醒。
    前一件事没写 until 时，按最宽松的算：他做完立刻动身（所以只有“怎么都来不及”才会提醒）。"""
    if c.net is None:
        return
    for pid, lst in tl.items():
        cur: Stay | None = None  # 当前这一站里最晚离开的那个事件
        leave = 0
        for s in lst:
            f = s.until if s.until is not None else s.start
            if cur is None:
                cur, leave = s, f
                continue
            if s.where == cur.where:
                if f >= leave:
                    cur, leave = s, f
                continue
            a, b, avail = cur.where, s.where, s.start - leave
            if avail >= 0 and a in c.locs and b in c.locs:
                res = c.net.shortest(a, b)
                head = (f"{s.ev.label}：{c.P(pid)} 从 {c.L(a)}（{cur.ev.label}，{fmt_time(leave)}）"
                        f"到 {c.L(b)}（{fmt_time(s.start)}）")
                if res is None:
                    c.r.add("walk_unreachable", f"{head}，links 里找不到连通这两处的路（可能是 links 漏写了）")
                elif res[0] > avail:
                    c.r.add("walk_short",
                            f"{head}，中间只有 {fmt_gap(avail)}，按 links 最快要走 {res[0]:g} 分钟"
                            f"（{' → '.join(res[1])}），时间不够")
            cur, leave = s, f


# ---------------------------------------------------------------- 剧本里的段号

@dataclass
class Script:
    path: str  # 显示用，比如 EP01/剧本.md
    segs: list[int]
    problem: str | None = None  # 读不了、认不出段标题时写原因


def parse_script(text: str) -> tuple[list[int], str | None]:
    """从剧本里认出段号。只看“## 正文”那一节，规则和 blind_pack.py 一样。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    m = BODY_RE.search(text)
    if not m:
        return [], "没有“## 正文”这一节"
    rest = text[m.end():]
    nxt = NEXT_H2_RE.search(rest)  # 正文之后如果还有别的二级标题，不算正文
    body = rest[: nxt.start()] if nxt else rest
    segs = [int(x.group(1)) for x in SEG_RE.finditer(body)]
    if not segs:
        return [], "正文里没有“**第 N 段 …**”这样的段标题"
    return segs, None


def load_scripts(show: Path, eps: Any) -> dict[int, Script | None]:
    out: dict[int, Script | None] = {}
    for n in sorted(set(eps)):
        out[n] = None
        for name in dict.fromkeys((f"EP{n:02d}", f"EP{n}")):
            p = show / name / "剧本.md"
            try:
                if not p.is_file():
                    continue
                segs, problem = parse_script(p.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError) as ex:
                out[n] = Script(f"{name}/剧本.md", [], f"读不了：{ex}")
                break
            out[n] = Script(f"{name}/剧本.md", segs, problem)
            break
    return out


def seg_refs(c: Ctx) -> list[tuple[int, int, str, str | None]]:
    """图谱里所有写了（集、段）的地方：(集, 段, 说明, 故事线)。"""
    refs: list[tuple[int, int, str, str | None]] = []
    for b in c.bts:
        if b.pos[1] is not None:
            refs.append((b.pos[0], b.pos[1], f"{b.label}：seg ", b.arc))
    for cl in c.cls:
        for fname, p in (("planted", cl.planted), ("payoff", cl.payoff), ("debunked", cl.debunked)):
            if p and p[1] is not None:
                refs.append((p[0], p[1], f"{cl.label}：{fname} ", cl.arc))
    for e in c.evs:
        if e.pos and e.pos[1] is not None:
            refs.append((e.pos[0], e.pos[1], f"{e.label}：ep、seg ", None))
    for lab, aid, p in c.clears:
        if p[1] is not None:
            refs.append((p[0], p[1], f"{lab} ", aid))
    return refs


def check_segments(c: Ctx, scripts: dict[int, Script | None]) -> None:
    """节拍、线索（planted、payoff、debunked）、事件、clearedAt 引用的（集、段），剧本里得有。"""
    refs = seg_refs(c)
    no_file: set[int] = set()
    bad_file: dict[int, str] = {}
    for ep, seg, what, arc in refs:
        sc = scripts.get(ep)
        if sc is None:
            no_file.add(ep)
        elif sc.problem:
            bad_file[ep] = f"{sc.path}：{sc.problem}"
        elif seg not in sc.segs:
            c.r.add("seg_missing",
                    f"{what}写的是{fmt_pos((ep, seg))}，但 {sc.path} 里没有这一段（剧本里有第 {fmt_ranges(sc.segs)} 段）", arc)
    for ep in sorted(no_file):
        c.r.add("no_script", f"第 {ep} 集没有剧本文件（找过 EP{ep:02d}/剧本.md），段号没核对")
    for ep in sorted(bad_file):
        c.r.add("script_unparsed", f"第 {ep} 集的剧本认不出段号（{bad_file[ep]}），段号没核对")


# ---------------------------------------------------------------- 线索

def is_whodunit(arc: dict) -> bool:
    return arc.get("type") in (None, "", "whodunit")  # 没写 type 也按 whodunit 算


def seg_counts(c: Ctx, scripts: dict[int, Script | None]) -> dict[int, int]:
    """每一集大概有几段：剧本里有就用剧本的，没有就用图谱里写到过的最大段号。"""
    n: dict[int, int] = {}

    def see(p: Pos | None) -> None:
        if p and p[1] is not None:
            n[p[0]] = max(n.get(p[0], 0), p[1])

    for e in c.evs:
        see(e.pos)
    for b in c.bts:
        see(b.pos)
    for cl in c.cls:
        for p in (cl.planted, cl.payoff, cl.debunked):
            see(p)
    for _lab, _aid, p in c.clears:
        see(p)
    for ep, sc in scripts.items():
        if sc and sc.segs and not sc.problem:
            n[ep] = max(n.get(ep, 0), max(sc.segs))
    return n


def arc_span(c: Ctx, aid: str) -> tuple[Pos, Pos] | None:
    """案子的开头（这条线最早的节拍）和揭晓（reveal 节拍；没有就取最后一个 payoff）。"""
    beats = [b for b in c.bts if b.arc == aid]
    if not beats:
        return None
    start = min((b.pos for b in beats), key=pos_key)
    revs = [b.pos for b in beats if b.kind == "reveal"]
    if revs:
        return start, max(revs, key=pos_key)
    pays = [cl.payoff for cl in c.cls if cl.arc == aid and cl.payoff]
    return (start, max(pays, key=pos_key)) if pays else None


def check_clues(c: Ctx, scripts: dict[int, Script | None]) -> None:
    r = c.r
    counts = seg_counts(c, scripts)

    def ordinal(p: Pos) -> int:  # 把（集、段）排成一条线上的第几段
        return sum(v for k, v in counts.items() if k < p[0]) + (p[1] or 0)

    by_arc: dict[str, list[Cl]] = {}
    for cl in c.cls:
        if cl.arc:
            by_arc.setdefault(cl.arc, []).append(cl)

    for cl in c.cls:
        arc = cl.arc
        if arc is None and blank(cl.item.get("arc")):
            r.add("clue_no_arc", f"{cl.label}：没写 arc（属于哪条故事线），所以查不了“太早暴露凶手”和“孤证”")
        if cl.payoff and not cl.planted:
            r.add("payoff_no_planted",
                  f"{cl.label}：有 payoff（{fmt_pos(cl.payoff)}用上），却没有 planted（观众第一次看到它的位置）", arc)
        elif cl.payoff and cl.planted and pos_cmp(cl.payoff, cl.planted) == -1:
            r.add("payoff_before_planted",
                  f"{cl.label}：用上（{fmt_pos(cl.payoff)}）早于埋下（{fmt_pos(cl.planted)}）", arc)
        if cl.debunked and cl.planted and pos_cmp(cl.debunked, cl.planted) == -1:
            r.add("debunk_early",
                  f"{cl.label}：被推翻（{fmt_pos(cl.debunked)}）早于埋下（{fmt_pos(cl.planted)}）", arc)
        if cl.real and cl.planted and not cl.payoff:
            r.add("unused_clue",
                  f"{cl.label}：埋在{fmt_pos(cl.planted)}，却没有 payoff（揭晓时没用上）；如果是留给后面的伏笔可以不管", arc)
        if not cl.real and not cl.debunked:
            r.add("herring_open", f"{cl.label}：是误导线索（real: false），却没写 debunked（在哪一段被推翻）", arc)

    for aid, arc in c.arcs.items():
        cul = c.culprits.get(aid, [])
        if not cul:
            continue
        clues = by_arc.get(aid, [])
        title = c.arc_title(aid)
        # 孤证：揭晓时用到的真线索里，指向凶手的只有一条
        pointing = [x for x in clues if x.real and x.payoff and set(x.points) & set(cul)]
        who_txt = "、".join(c.P(p) for p in cul)
        if len(pointing) == 1:
            r.add("single_proof",
                  f"{title}：揭晓时用到的真线索里，只有一条指向凶手 {who_txt}，就是{pointing[0].label}（孤证）", aid)
        elif not pointing:
            r.add("no_proof",
                  f"{title}：揭晓时用到的真线索（有 payoff 的）里，没有一条指向凶手 {who_txt}，观众没法靠线索推出来", aid)
        # 可能太早暴露凶手：只查 whodunit
        if not is_whodunit(arc):
            continue
        span = arc_span(c, aid)
        if span is None or span[0][1] is None or span[1][1] is None:
            continue
        start, reveal = span
        o_s, o_r = ordinal(start), ordinal(reveal)
        if o_r <= o_s:
            continue
        for cl in clues:
            if not cl.real or not cl.planted or cl.planted[1] is None:
                continue
            target = set(cl.points)
            if len(target) != 1 or next(iter(target)) not in cul:
                continue
            who = next(iter(target))
            o_p = ordinal(cl.planted)
            if (o_p - o_s) * 3 > (o_r - o_s) or o_p >= o_r:
                continue  # 不在前三分之一（用整数乘法比，免得小数误差）
            frac = max(0.0, (o_p - o_s) / (o_r - o_s))
            where = "案子最开头" if frac == 0 else f"案子开头到揭晓的前 1/{max(3, round(1 / frac))}"
            cover = next((m for m in clues
                          if not m.real and m.planted and set(m.points) - {who}
                          and pos_cmp(m.planted, cl.planted) == -1
                          and (m.debunked is None or pos_cmp(cl.planted, m.debunked) == -1)), None)
            span_txt = f"案子 {aid} 从{fmt_pos(start)}开始、{fmt_pos(reveal)}揭晓"
            if cover is None:
                r.add("early_exposure",
                      f"{cl.label}：可能太早暴露凶手（真线索只指向 {c.P(who)}，埋在{fmt_pos(cl.planted)}，"
                      f"位于{where}；{span_txt}）", aid)
            else:
                others = "、".join(c.P(p) for p in cover.points if p != who)
                until = f"，要到{fmt_pos(cover.debunked)}才被推翻" if cover.debunked else "，一直没被推翻"
                r.add("early_exposure_light",
                      f"{cl.label}：（较轻）真线索只指向 {c.P(who)}、埋在{fmt_pos(cl.planted)}，位置很早（{where}；{span_txt}），"
                      f"不过这时{cover.label}（误导线索）还在把人引向 {others}{until}，影响不大，供参考", aid)


# ---------------------------------------------------------------- 泄底

def before_reveal(p: Pos, reveal: Pos | None) -> bool:
    """这个位置是不是在揭晓之前。揭晓还没写，就当作还没揭晓；同一集里有一边没写段，比不出来就不算。"""
    return True if reveal is None else pos_cmp(p, reveal) == -1


def find_name(names: dict[str, str], *texts: Any) -> str | None:
    blob = "\n".join(t for t in texts if isinstance(t, str))
    return next((nm for nm in names if nm in blob), None)


def check_spoilers(c: Ctx) -> None:
    r = c.r
    cul_arcs: dict[str, list[str]] = {}  # 凶手 -> 他是哪些（whodunit）案子的凶手
    for aid, arc in c.arcs.items():
        if is_whodunit(arc):
            for p in c.culprits.get(aid, []):
                cul_arcs.setdefault(p, []).append(aid)
    # 没上屏的事件（没写 ep、seg）：牵涉凶手的必须标 spoiler，别的最好也标
    for e in c.evs:
        if not e.offscreen or e.spoiler:
            continue
        hit = [p for p in e.who if p in cul_arcs]
        what = f"（what：「{snippet(e.item.get('what', ''), 20)}」）"
        if hit:
            r.add("spoiler_must",
                  f"{e.label}：没上屏（没写 ep、seg）的事件牵涉凶手 {c.P(hit[0])}，却没标 spoiler: true{what}"
                  f"，画布在观众视角下会把底泄了", cul_arcs[hit[0]][0])
        else:
            r.add("spoiler_warn",
                  f"{e.label}：没上屏（没写 ep、seg）的事件，观众此刻不知道，建议标 spoiler: true{what}")
    # 节拍和线索的文字里写了凶手的名字，又没标 spoiler，而且在揭晓之前
    for aid, arc in c.arcs.items():
        cul = c.culprits.get(aid, [])
        if not cul or not is_whodunit(arc):
            continue
        names: dict[str, str] = {}
        for p in cul:
            ch = c.chars.get(p) or {}
            for key in ("name", "nameKo"):
                v = ch.get(key)
                if isinstance(v, str) and len(v.strip()) >= 2:
                    names[v.strip()] = p
        if not names:
            continue
        span = arc_span(c, aid)
        reveal = span[1] if span else None
        when = f"揭晓在{fmt_pos(reveal)}" if reveal else "这条线还没写揭晓"
        for b in c.bts:
            if b.arc != aid or b.spoiler or not before_reveal(b.pos, reveal):
                continue
            nm = find_name(names, b.text)
            if nm:
                r.add("name_leak",
                      f"{b.label}：text 里出现了凶手 {c.P(names[nm])} 的名字「{nm}」，又没标 spoiler，"
                      f"位置在揭晓之前（{fmt_pos(b.pos)}；{when}），请人工看一眼会不会泄底", aid)
        for cl in c.cls:
            if cl.arc != aid or cl.spoiler or not cl.planted or not before_reveal(cl.planted, reveal):
                continue
            nm = find_name(names, cl.item.get("name"), cl.item.get("text"))
            if nm:
                r.add("name_leak",
                      f"{cl.label}：name 或 text 里出现了凶手 {c.P(names[nm])} 的名字「{nm}」，又没标 spoiler，"
                      f"埋在揭晓之前（{fmt_pos(cl.planted)}；{when}），请人工看一眼会不会泄底", aid)


# ---------------------------------------------------------------- 总调度、输出

def run_checks(data: dict[str, Any], show: Path | None = None,
               scripts: dict[int, Script | None] | None = None) -> tuple[Ctx, Report]:
    r = Report()
    c = Ctx(data, r)
    if scripts is None:
        need = set(c.eps) | {ep for ep, _seg, _what, _arc in seg_refs(c)}
        scripts = load_scripts(show, need) if show is not None else {}
    tl = build_timelines(c)
    check_two_places(c, tl)
    check_learns(c)
    check_perceives(c, tl)
    check_walks(c, tl)
    check_clues(c, scripts)
    check_spoilers(c)
    check_segments(c, scripts)
    return c, r


def breakdown(issues: list[Issue]) -> str:
    """“必改 2（一人两地 1、…）· 提醒 3（…）”"""
    parts = []
    for level in (MUST, WARN):
        xs = [x for x in issues if x.level == level]
        if not xs:
            parts.append(f"{level} 0")
            continue
        kinds = [f"{lab} {n}" for code, (lv, lab) in CODES.items()
                 if lv == level and (n := sum(1 for x in xs if x.code == code))]
        parts.append(f"{level} {len(xs)}（{'、'.join(kinds)}）")
    return " · ".join(parts)


def render(c: Ctx, r: Report) -> str:
    title = c.data.get("title")
    out = [f"故事图谱检查 · {title if isinstance(title, str) and title.strip() else '（没写剧名）'}"]
    for mark, level in (("✗", MUST), ("⚠", WARN)):
        xs = r.by_level(level)
        if not xs:
            out.append(f"{mark} {level}（0）：没有")
            continue
        out.append(f"{mark} {level}（{len(xs)}）")
        out.extend(f"  - {x.text}" for x in xs)
    # 小计
    arcs_txt = []
    for aid, arc in c.arcs.items():
        clues = [x for x in c.cls if x.arc == aid]
        if clues:
            real = sum(1 for x in clues if x.real)
            nm = arc.get("name")
            arcs_txt.append(f"{short_arc_name(nm) if isinstance(nm, str) and nm.strip() else aid}：真 {real} / 误导 {len(clues) - real}")
    loose = sum(1 for x in c.cls if x.arc not in c.arcs)
    if loose:
        arcs_txt.append(f"没归到任何故事线：{loose}")
    out.append(
        f"✓ 小计：人物 {len(c.coll['characters'])} · 地点 {len(c.coll['locations'])} · 事件 {len(c.coll['events'])}"
        f" · 线索 {len(c.coll['clues'])}" + (f"（{'；'.join(arcs_txt)}）" if arcs_txt else ""))
    out.append(f"  问题合计：{breakdown(r.issues)}")
    if c.arcs:
        out.append("  各条故事线：")
        for aid, arc in c.arcs.items():
            clues = [x for x in c.cls if x.arc == aid]
            real = sum(1 for x in clues if x.real)
            mine = [x for x in r.issues if x.arc == aid]
            nm = arc.get("name")
            out.append(f"  - {nm if isinstance(nm, str) and nm.strip() else aid}（{aid}）：真线索 {real} · 误导 {len(clues) - real}"
                       f" ｜ {breakdown(mine) if mine else '没有问题'}")
        rest = [x for x in r.issues if x.arc not in c.arcs]
        if rest:
            out.append(f"  - 不属于某条故事线的问题：{breakdown(rest)}")
    n_must = len(r.by_level(MUST))
    out.append(f"→ 有 {n_must} 条必改，改完再跑一遍。" if n_must else "→ 没有必改。提醒只是请人看一眼，不一定要改。")
    return "\n".join(out)


def check_show(show: Path) -> tuple[str, int]:
    """读一部剧的文件夹，返回（要打印的文字，退出码）。"""
    path = show / "故事图谱.json"
    try:
        is_dir = show.is_dir()
    except OSError as ex:
        return f"✗ 读不了剧的文件夹 {show}：{ex}", 1
    if not is_dir:
        return f"✗ 找不到剧的文件夹：{show}", 1
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return f"✗ 找不到 {path}（剧的文件夹里要有 故事图谱.json）", 1
    except (OSError, UnicodeDecodeError) as ex:
        return f"✗ 读不了 {path}：{ex}", 1
    except json.JSONDecodeError as ex:
        return f"✗ {path.name} 不是合法的 JSON：第 {ex.lineno} 行第 {ex.colno} 列，{ex.msg}", 1
    if not isinstance(data, dict):
        return f"✗ {path.name} 最外层应该是一个对象 {{…}}，现在是{tname(data)}", 1
    c, r = run_checks(data, show=show)
    return render(c, r), 1 if r.by_level(MUST) else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="检查故事图谱有没有逻辑漏洞。有“必改”时退出码为 1。")
    ap.add_argument("show", nargs="?", type=Path, help="剧的文件夹，里面有 故事图谱.json（可选 EP01/剧本.md）")
    ap.add_argument("--selftest", action="store_true", help="用自带的小样本自测，不联网、不调用模型")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.show is None:
        ap.error("要给剧的文件夹，或者用 --selftest")
    try:
        text, code = check_show(a.show.expanduser())
    except Exception:  # noqa: BLE001  脚本自己出错时不能冒充“有必改”（退出码 1），单独用 2
        import traceback
        traceback.print_exc()
        print("✗ 检查脚本自己出错了（不是图谱的问题），请把上面的报错告诉写脚本的人。", file=sys.stderr)
        return 2
    print(text)
    return code


# ---------------------------------------------------------------- 自测：样本

def _ev(eid: str, at: str, who: list[str], where: str, what: str, **kw: Any) -> dict[str, Any]:
    return {"id": eid, "at": at, "who": who, "where": where, "what": what, **kw}


def _good() -> tuple[dict[str, Any], dict[int, Script | None]]:
    """一份没有任何问题的小图谱（和它对应的剧本段号）。好样本不能报出任何一条。"""
    g: dict[str, Any] = {
        "version": 1,
        "title": "自测好样本",
        "characters": [
            {"id": "ann", "name": "安安", "group": "主角"},
            {"id": "bo", "name": "波波"},
            {"id": "kai", "name": "凯总", "arc": "case1"},
            {"id": "dee", "name": "迪迪", "arc": "case2"},
        ],
        "relations": [{"from": "ann", "to": "bo", "label": "同事", "kind": "work", "arc": "case1"}],
        "maps": [{"id": "inn", "name": "小旅馆", "grid": [6, 2],
                  "layers": [{"id": "f2", "name": "二楼"}, {"id": "f1", "name": "一楼"}]}],
        "locations": [
            {"id": "lobby", "name": "前台", "map": "inn", "layer": "f1"},
            {"id": "yard", "name": "院子", "map": "inn", "layer": "f1"},
            {"id": "hall", "name": "走廊", "map": "inn", "layer": "f2"},
            {"id": "r1", "name": "一号房", "map": "inn", "layer": "f2"},
            {"id": "r2", "name": "二号房", "map": "inn", "layer": "f2"},
        ],
        "links": [
            {"from": "lobby", "to": "hall", "via": "楼梯", "walk": 1, "see": True, "hear": True},
            {"from": "hall", "to": "r1", "via": "门", "walk": 0, "see": False, "hear": True},
            {"from": "hall", "to": "r2", "via": "门", "walk": 1, "see": False, "hear": False},
            {"from": "lobby", "to": "yard", "via": "后门", "walk": 2, "see": True, "hear": True},
        ],
        "arcs": [
            {"id": "case1", "name": "案子1 · 旅馆敲墙", "kind": "case", "type": "whodunit", "episodes": [1, 1],
             "culprit": ["kai"], "truth": {"text": "凯总躲在一号房敲墙", "spoiler": True},
             "suspects": [{"who": "dee", "motive": "想吓走客人", "clearedAt": {"ep": 1, "seg": 8}}]},
            {"id": "case2", "name": "案子2 · 看怎么抓", "kind": "case", "type": "howcatch", "episodes": [2, 2],
             "culprit": ["dee"]},
        ],
        "episodes": [
            {"n": 1, "title": "第一集", "status": "待查", "beats": [
                {"arc": "case1", "text": "安安住进小旅馆", "seg": 1, "kind": "setup"},
                {"arc": "case1", "text": "凯总悄悄回到旅馆", "seg": 3, "kind": "turn", "spoiler": True},
                {"arc": "case1", "text": "半夜有人敲墙", "seg": 4, "kind": "hook"},
                {"arc": "case1", "text": "凯总被当场抓住", "seg": 10, "kind": "reveal"},
            ]},
            {"n": 2, "title": "第二集", "beats": [
                {"arc": "case2", "text": "迪迪一出场就可疑", "seg": 1, "kind": "setup"},
                {"arc": "case2", "text": "抓住迪迪", "seg": 5, "kind": "reveal"},
            ]},
        ],
        "clues": [
            {"id": "c_knock", "arc": "case1", "name": "敲墙声", "planted": {"ep": 1, "seg": 5},
             "payoff": {"ep": 1, "seg": 10}, "pointsTo": ["kai"], "real": True},
            {"id": "c_coat", "arc": "case1", "name": "湿外套", "planted": {"ep": 1, "seg": 6},
             "payoff": {"ep": 1, "seg": 10}, "pointsTo": ["kai", "dee"], "real": True},
            {"id": "c_herring", "arc": "case1", "name": "怪笑", "planted": {"ep": 1, "seg": 3},
             "pointsTo": ["dee"], "real": False, "debunked": {"ep": 1, "seg": 8}},
            {"id": "c_d1", "arc": "case2", "name": "手套", "planted": {"ep": 2, "seg": 1},
             "payoff": {"ep": 2, "seg": 5}, "pointsTo": ["dee"], "real": True},
            {"id": "c_d2", "arc": "case2", "name": "脚印", "planted": {"ep": 2, "seg": 2},
             "payoff": {"ep": 2, "seg": 5}, "pointsTo": ["dee"], "real": True},
        ],
        "events": [
            _ev("e1", "D1 18:00", ["ann"], "lobby", "安安在前台登记", until="D1 18:30", ep=1, seg=1,
                learns=[{"who": "ann", "fact": "旅馆半夜闹鬼", "from": "老板娘当面说的"},
                        {"who": "audience", "fact": "安安听说闹鬼", "from": "老板娘的台词"}]),
            _ev("e2", "D1 18:31", ["ann"], "hall", "安安上楼到走廊", ep=1, seg=2),
            _ev("e3", "D1 21:50", ["dee"], "hall", "迪迪在走廊擦地", until="D1 22:10", ep=1, seg=3),
            _ev("e4", "D1 22:00", ["kai"], "r1", "凯总躲在一号房，贴着墙敲了三下", spoiler=True,
                perceives=[{"who": "dee", "how": "听见", "what": "墙里三声"},
                           {"who": "ann", "how": "闻到", "what": "烟味"}]),
            _ev("e5", "D1 22:05", ["ann"], "r2", "安安回二号房", ep=1, seg=4),
            _ev("e6", "D1 22:30", ["kai"], "lobby", "凯总溜到前台", ep=1, seg=5),
            _ev("e7", "D1 22:30", ["dee"], "yard", "迪迪在院子抽烟", until="D1 22:40", ep=1, seg=6),
            _ev("e8", "D1 23:00", ["ann"], "lobby", "安安回到前台", until="D1 23:10", ep=1, seg=7,
                perceives=[{"who": "bo", "how": "看见", "what": "安安站在前台"}]),
            _ev("e9", "D1 23:00", ["bo"], "yard", "波波在院子里打电话", until="D1 23:20", ep=1, seg=7),
            _ev("e10", "D1 23:30", ["ann"], "hall", "安安去走廊", until="D1 23:40", ep=1, seg=8),
            _ev("e11", "D1 23:40", ["ann"], "r1", "安安进一号房（和上一件首尾相接）", until="D1 23:45", ep=1, seg=8),
            _ev("e12", "D1 20:00", ["bo"], "lobby", "波波在前台（没写 until）", ep=1, seg=2),
            _ev("e13", "D1 20:10", ["bo"], "r2", "波波回二号房（十分钟后，换了地方）", ep=1, seg=3),
        ],
    }
    scripts: dict[int, Script | None] = {
        1: Script("EP01/剧本.md", list(range(1, 11))),
        2: Script("EP02/剧本.md", list(range(1, 6))),
    }
    return g, scripts


def _walk_path(obj: Any, path: tuple) -> Any:
    for k in path:
        if isinstance(obj, list) and isinstance(k, str):
            obj = next(x for x in obj if isinstance(x, dict) and x.get("id") == k)  # 列表里按 id 找
        else:
            obj = obj[k]
    return obj


def _apply(g: dict[str, Any], scripts: dict[int, Script | None], ops: list[tuple]) -> None:
    """改样本的小语言：("set", 路径, 值) / ("del", 路径) / ("add", 列表路径, 值) / ("script", 集, 值)。"""
    for op in ops:
        if op[0] == "script":
            scripts[op[1]] = op[2]
            continue
        if op[0] == "add":
            _walk_path(g, op[1]).append(copy.deepcopy(op[2]))
            continue
        parent, last = _walk_path(g, op[1][:-1]), op[1][-1]
        if isinstance(parent, list) and isinstance(last, str):
            last = next(i for i, x in enumerate(parent) if x.get("id") == last)
        if op[0] == "set":
            parent[last] = copy.deepcopy(op[2])
        else:
            del parent[last]


# 每一项：(名字, 改法, 应该查出的种类, 允许顺带出现的种类)。种类为空 = 这样改完不该报任何东西。
_X = frozenset()
_CASES: list[tuple[str, list[tuple], set[str], set[str]]] = [
    # ---- 基础
    ("缺必填：人物没有 name", [("del", ("characters", "bo", "name"))], {"missing_field"}, _X),
    ("缺必填：事件没有 what", [("del", ("events", "e3", "what"))], {"missing_field"}, _X),
    ("缺必填：最上面没有 version", [("del", ("version",))], {"missing_field"}, _X),
    ("缺必填：节拍没有 text", [("del", ("episodes", 0, "beats", 2, "text"))], {"missing_field"}, _X),
    ("缺必填：事件 who 是空列表", [("set", ("events", "e7", "who"), [])], {"missing_field"}, _X),
    ("id 重复：人物", [("add", ("characters",), {"id": "ann", "name": "另一个安安"})], {"dup_id"}, _X),
    ("id 重复：事件", [("add", ("events",), _ev("e1", "D2 09:00", ["bo"], "yard", "重名事件", ep=1, seg=2))], {"dup_id"}, _X),
    ("id 重复：地点", [("add", ("locations",), {"id": "hall", "name": "又一个走廊", "map": "inn", "layer": "f2"})], {"dup_id"}, _X),
    ("id 重复：线索", [("add", ("clues",), {"id": "c_herring", "arc": "case1", "name": "重名", "real": False,
                                          "debunked": {"ep": 1, "seg": 8}})], {"dup_id"}, _X),
    ("id 重复：集", [("add", ("episodes",), {"n": 1, "title": "又一个第 1 集"})], {"dup_id"}, _X),
    ("id 重复：同一张地图里的楼层", [("add", ("maps", "inn", "layers"), {"id": "f1", "name": "又一个一楼"})], {"dup_id"}, _X),
    ("引用不存在：事件的地点", [("set", ("events", "e2", "where"), "nowhere")], {"bad_ref"}, _X),
    ("引用不存在：事件里的人", [("set", ("events", "e1", "who"), ["ann", "ghost"])], {"bad_ref"}, _X),
    ("引用不存在：perceives 里的人", [("add", ("events", "e4", "perceives"), {"who": "ghost", "how": "听见", "what": "x"})], {"bad_ref"}, _X),
    ("引用不存在：learns 里的人", [("add", ("events", "e1", "learns"), {"who": "ghost", "fact": "x", "from": "y"})], {"bad_ref"}, _X),
    ("引用不存在：线索的故事线", [("set", ("clues", "c_herring", "arc"), "case9")], {"bad_ref"}, _X),
    ("引用不存在：线索指向的人", [("set", ("clues", "c_herring", "pointsTo"), ["ghost"])], {"bad_ref"}, _X),
    ("引用不存在：节拍的故事线", [("set", ("episodes", 0, "beats", 2, "arc"), "case9")], {"bad_ref"}, _X),
    ("引用不存在：关系里的人", [("set", ("relations", 0, "from"), "ghost")], {"bad_ref"}, _X),
    ("引用不存在：凶手", [("set", ("arcs", "case1", "culprit"), ["kai", "ghost"])], {"bad_ref"}, _X),
    ("引用不存在：嫌疑人", [("set", ("arcs", "case1", "suspects", 0, "who"), "ghost")], {"bad_ref"}, _X),
    ("引用不存在：人物的故事线", [("set", ("characters", "kai", "arc"), "case9")], {"bad_ref"}, _X),
    ("引用不存在：地点的地图", [("set", ("locations", "r2", "map"), "ghostmap")], {"bad_ref"}, _X),
    ("引用不存在：地点的楼层", [("set", ("locations", "r2", "layer"), "f9")], {"bad_ref"}, _X),
    ("引用不存在：links 的地点", [("add", ("links",), {"from": "hall", "to": "nowhere", "walk": 1})], {"bad_ref"}, _X),
    ("没写 map 只写 layer（楼层在某张地图里）：不报", [("del", ("locations", "r2", "map"))], set(), _X),
    ("时间格式：D1 6pm", [("set", ("events", "e1", "at"), "D1 6pm")], {"bad_time"}, _X),
    ("时间格式：25 点", [("set", ("events", "e9", "until"), "D1 25:00")], {"bad_time"}, _X),
    ("时间格式：小写 d", [("set", ("events", "e12", "at"), "d1 20:00")], {"bad_time"}, _X),
    ("时间格式：秒写成 60", [("set", ("events", "e9", "until"), "D1 23:20:60")], {"bad_time"}, _X),
    ("时间格式：秒只写一位", [("set", ("events", "e12", "at"), "D1 20:00:5")], {"bad_time"}, _X),
    ("until 早于 at", [("set", ("events", "e7", "until"), "D1 22:00")], {"until_before_at"}, _X),
    # ---- 一人两地
    ("一人两地：时间重叠、地点不同", [("add", ("events",), _ev("e30", "D1 18:10", ["ann"], "r2", "重叠的事", until="D1 18:20", ep=1, seg=2))],
     {"two_places"}, _X),
    ("一人两地：跨过午夜也能比", [("set", ("events", "e10", "until"), "D2 00:10"), ("set", ("events", "e10", "where"), "hall"),
                        ("add", ("events",), _ev("e31", "D2 00:05", ["ann"], "lobby", "午夜后的重叠", until="D2 00:20", ep=1, seg=8))],
     {"two_places"}, {"walk_short"}),
    ("重叠但同一个地点：不报", [("add", ("events",), _ev("e32", "D1 18:10", ["ann"], "lobby", "同地点的另一件事", until="D1 18:20", ep=1, seg=2))], set(), _X),
    ("同一分钟、没写 until、换了地方：不算重叠", [("add", ("events",), _ev("e33", "D1 18:31", ["ann"], "r1", "同一分钟", ep=1, seg=2))], set(), _X),
    ("同一分钟连着写好几件事（没写秒）、换来换去：按列表顺序当先后，不算重叠",
     [("add", ("events",), _ev("e37", "D1 20:00", ["ann"], "hall", "安安到走廊", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:00", ["ann"], "r1", "安安推开一号房的门", ep=1, seg=3)),
      ("add", ("events",), _ev("e39", "D1 20:00", ["ann"], "hall", "安安退回走廊", ep=1, seg=3))], set(), _X),
    ("同一分钟里写了秒、先后换了地方：不算重叠",
     [("add", ("events",), _ev("e37", "D1 20:00:00", ["ann"], "hall", "安安到走廊", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:00:04", ["ann"], "r1", "安安推开一号房的门", ep=1, seg=3)),
      ("add", ("events",), _ev("e39", "D1 20:00:45", ["ann"], "hall", "安安退回走廊", ep=1, seg=3))], set(), _X),
    ("一人两地：带秒的时间块重叠了一秒",
     [("add", ("events",), _ev("e37", "D1 20:00:00", ["ann"], "hall", "安安在走廊", until="D1 20:00:30", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:00:29", ["ann"], "r2", "安安在二号房", until="D1 20:00:45", ep=1, seg=3))],
     {"two_places"}, _X),
    ("带秒的时间块首尾相接：不报",
     [("add", ("events",), _ev("e37", "D1 20:00:00", ["ann"], "hall", "安安在走廊", until="D1 20:00:30", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:00:30", ["ann"], "r1", "安安在一号房", until="D1 20:00:45", ep=1, seg=3))], set(), _X),
    ("一人两地：零长度的事件落在别人的时间中间",
     [("add", ("events",), _ev("e37", "D1 20:00", ["ann"], "hall", "安安在走廊", until="D1 20:10", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:05", ["ann"], "r1", "安安探头看一眼一号房", ep=1, seg=3)),
      ("add", ("events",), _ev("e39", "D1 20:05", ["ann"], "hall", "又回到走廊", ep=1, seg=3))], {"two_places"}, _X),
    # ---- 知道要有来源
    ("learns 缺 from", [("del", ("events", "e1", "learns", 0, "from"))], {"learns_no_from"}, _X),
    ("learns 的 from 是空字符串", [("set", ("events", "e1", "learns", 0, "from"), "")], {"learns_no_from"}, _X),
    # ---- 看见听见
    ("听见：links 里 hear 没标 true", [("set", ("links", 1, "hear"), False)], {"perceive_far"}, _X),
    ("看见：links 里 see 没标 true", [("set", ("links", 3, "see"), False)], {"perceive_far"}, _X),
    ("听见：他当时在不相邻的地点", [("set", ("events", "e13", "at"), "D1 21:50"),
                           ("add", ("events", "e4", "perceives"), {"who": "bo", "how": "听见", "what": "墙里三声"})],
     {"perceive_far"}, _X),
    ("听见：不知道他那一刻在哪（没写 until 的事件只算 15 分钟）",
     [("add", ("events", "e4", "perceives"), {"who": "bo", "how": "听见", "what": "墙里三声"})], {"perceive_unsure"}, _X),
    ("文件里没有 links：只查同一个地点，其余算提醒", [("del", ("links",))], {"perceive_unsure"}, _X),
    # ---- 路
    ("路：时间不够走过去", [("add", ("events",), _ev("e34", "D1 18:32", ["ann"], "yard", "很快到了院子", ep=1, seg=2))], {"walk_short"}, _X),
    ("路：两处根本连不通", [("add", ("locations",), {"id": "island", "name": "孤岛", "map": "inn", "layer": "f1"}),
                      ("add", ("events",), _ev("e35", "D1 19:00", ["ann"], "island", "去了孤岛", ep=1, seg=2))], {"walk_unreachable"}, _X),
    ("路：带秒的间隔太短（4 秒走不完 1 分钟的路）",
     [("add", ("events",), _ev("e37", "D1 20:00:00", ["ann"], "lobby", "安安在前台", ep=1, seg=3)),
      ("add", ("events",), _ev("e38", "D1 20:00:04", ["ann"], "hall", "安安已经在走廊了", ep=1, seg=3))], {"walk_short"}, _X),
    ("路：walk 不写当 1 分钟（刚好够）：不报", [("del", ("links", 0, "walk"))], set(), _X),
    ("路：walk 改成 2 分钟（只隔 1 分钟，不够）", [("set", ("links", 0, "walk"), 2)], {"walk_short"}, _X),
    # ---- 线索
    ("线索：有 payoff 没有 planted", [("del", ("clues", "c_coat", "planted"))], {"payoff_no_planted"}, _X),
    ("线索：没写 arc", [("del", ("clues", "c_herring", "arc"))], {"clue_no_arc"}, _X),
    ("线索：用上早于埋下", [("set", ("clues", "c_knock", "payoff"), {"ep": 1, "seg": 3})], {"payoff_before_planted"}, _X),
    ("线索：太早暴露凶手", [("set", ("clues", "c_knock", "planted"), {"ep": 1, "seg": 2})], {"early_exposure"}, _X),
    ("线索：太早暴露凶手但有误导线索挡着（较轻）", [("set", ("clues", "c_knock", "planted"), {"ep": 1, "seg": 4})], {"early_exposure_light"}, _X),
    ("线索：误导线索刚好在这一段被推翻，不算挡着", [("set", ("clues", "c_knock", "planted"), {"ep": 1, "seg": 4}),
                                    ("set", ("clues", "c_herring", "debunked"), {"ep": 1, "seg": 4})], {"early_exposure"}, _X),
    ("线索：前三分之一以外不报（刚过一点点）", [("set", ("clues", "c_knock", "planted"), {"ep": 1, "seg": 5})], set(), _X),
    ("线索：案子跨两集，按整条线算前三分之一（埋在第 1 集，算早）",
     [("set", ("episodes", 0, "beats", 3, "kind"), "turn"), ("set", ("episodes", 0, "beats", 3, "spoiler"), True),
      ("add", ("episodes", 1, "beats"), {"arc": "case1", "text": "收网", "seg": 5, "kind": "reveal"})],
     {"early_exposure_light"}, _X),
    ("线索：案子跨两集，埋在第 2 集开头就不算早",
     [("set", ("episodes", 0, "beats", 3, "kind"), "turn"), ("set", ("episodes", 0, "beats", 3, "spoiler"), True),
      ("add", ("episodes", 1, "beats"), {"arc": "case1", "text": "收网", "seg": 5, "kind": "reveal"}),
      ("set", ("clues", "c_knock", "planted"), {"ep": 2, "seg": 1}), ("set", ("clues", "c_knock", "payoff"), {"ep": 2, "seg": 3})],
     set(), _X),
    ("线索：没写 type 也按 whodunit 查", [("del", ("arcs", "case2", "type"))], {"early_exposure"}, {"name_leak"}),
    ("线索：埋了没用上（顺带成了孤证）", [("del", ("clues", "c_coat", "payoff"))], {"unused_clue", "single_proof"}, _X),
    ("线索：误导没推翻", [("del", ("clues", "c_herring", "debunked"))], {"herring_open"}, _X),
    ("线索：被推翻早于埋下", [("set", ("clues", "c_herring", "debunked"), {"ep": 1, "seg": 2})], {"debunk_early"}, _X),
    ("线索：孤证", [("set", ("clues", "c_coat", "pointsTo"), ["dee"])], {"single_proof"}, _X),
    ("线索：没有一条指向凶手", [("set", ("clues", "c_coat", "pointsTo"), ["dee"]), ("set", ("clues", "c_knock", "pointsTo"), ["dee"])],
     {"no_proof"}, _X),
    # ---- 泄底
    ("泄底：没上屏的事件牵涉凶手却没标 spoiler", [("del", ("events", "e4", "spoiler"))], {"spoiler_must"}, _X),
    ("泄底：没上屏的事件没标 spoiler（不牵涉凶手）", [("add", ("events",), _ev("e36", "D1 12:00", ["bo"], "yard", "波波晒太阳"))], {"spoiler_warn"}, _X),
    ("泄底：节拍文字里有凶手名字", [("set", ("episodes", 0, "beats", 2, "text"), "凯总在走廊露了脸")], {"name_leak"}, _X),
    ("泄底：线索名字里有凶手名字", [("set", ("clues", "c_coat", "name"), "凯总的湿外套")], {"name_leak"}, _X),
    ("泄底：标了 spoiler 的不提醒", [("set", ("clues", "c_coat", "name"), "凯总的湿外套"), ("set", ("clues", "c_coat", "spoiler"), True)], set(), _X),
    # ---- 段号对剧本
    ("段号：节拍的段剧本里没有", [("set", ("episodes", 0, "beats", 2, "seg"), 11)], {"seg_missing"}, _X),
    ("段号：线索的段剧本里没有", [("set", ("clues", "c_knock", "payoff"), {"ep": 1, "seg": 12})], {"seg_missing"}, _X),
    ("段号：事件的段剧本里没有", [("set", ("events", "e1", "seg"), 12)], {"seg_missing"}, _X),
    ("段号：嫌疑人的 clearedAt 剧本里没有", [("set", ("arcs", "case1", "suspects", 0, "clearedAt"), {"ep": 1, "seg": 12})], {"seg_missing"}, _X),
    ("段号：节拍没写 seg，不核对", [("del", ("episodes", 0, "beats", 2, "seg"))], set(), _X),
    ("段号：某一集没有剧本文件（只提醒一次）", [("script", 2, None)], {"no_script"}, _X),
    ("段号：剧本认不出段标题（只提醒一次）", [("script", 2, Script("EP02/剧本.md", [], "没有“## 正文”这一节"))], {"script_unparsed"}, _X),
]

# 把各种错一起放进一份“坏样本”，每一类都要查得出来（下面这些改法互不挡路）
_BAD_PICK = [
    "缺必填：人物没有 name", "id 重复：人物", "引用不存在：事件里的人", "引用不存在：线索的故事线", "时间格式：25 点",
    "until 早于 at", "一人两地：时间重叠、地点不同", "learns 缺 from", "听见：links 里 hear 没标 true",
    "看见：links 里 see 没标 true", "路：时间不够走过去", "路：两处根本连不通", "线索：有 payoff 没有 planted",
    "线索：用上早于埋下", "线索：误导没推翻", "线索：孤证", "泄底：没上屏的事件牵涉凶手却没标 spoiler",
    "泄底：没上屏的事件没标 spoiler（不牵涉凶手）", "泄底：节拍文字里有凶手名字", "段号：事件的段剧本里没有",
    "段号：某一集没有剧本文件（只提醒一次）",
]


# ---------------------------------------------------------------- 自测：运行

def selftest() -> int:
    import contextlib
    import io

    passed = 0
    fails: list[str] = []

    def ok(cond: Any, name: str, detail: str = "") -> bool:
        nonlocal passed
        if cond:
            passed += 1
            return True
        fails.append(f"  ✗ {name}" + (f"\n      {detail}" if detail else ""))
        return False

    def brief(r: Report, codes: set[str]) -> str:
        return "\n      ".join(f"[{x.code}] {x.text}" for x in r.issues if x.code in codes)

    print("自测开始（不联网、不调用模型）")

    # ---- 小工具
    ok(parse_time("D2 02:05") == 2 * 1440 + 125 and parse_time(" D1 7:30 ") == 1440 + 450, "时间解析：正常写法")
    ok(all(parse_time(x) is None for x in ("D1 24:00", "D1 12:60", "d1 12:00", "D1 12", "第2天 02:00", "", None, 5)),
       "时间解析：错的写法都认不出来")
    ok(fmt_time(2 * 1440 + 125) == "D2 02:05" and fmt_span(1440, 1450) == "D1 00:00–00:10", "时间显示")
    ok(parse_time("D2 02:00:43") == 2 * 1440 + 120 + Fraction(43, 60) and parse_time("D2 02:00:00") == parse_time("D2 02:00") == 2 * 1440 + 120
       and parse_time(" D1 7:30:05 ") == 1440 + 450 + Fraction(5, 60), "时间解析：可以带秒（不写秒就是 :00）")
    ok(all(parse_time(x) is None for x in ("D2 02:00:60", "D2 02:00:5", "D2 02:00:", "D2 02:00:43:01", "D2 24:00:00", "D2 02:00 43")),
       "时间解析：秒写错的写法认不出来")
    ok(parse_time("D1 10:02:02") + 15 == parse_time("D1 10:17:02") and parse_time("D1 00:00:17") + 1 == parse_time("D1 00:01:17"),
       "时间相加：带秒的分钟数没有浮点误差")
    ok(fmt_time(parse_time("D2 02:00:43")) == "D2 02:00:43" and fmt_time(parse_time("D2 02:00:00")) == "D2 02:00"
       and fmt_span(parse_time("D1 00:00:10"), parse_time("D1 00:01:05")) == "D1 00:00:10–00:01:05"
       and fmt_span(parse_time("D1 23:59:50"), parse_time("D2 00:00:10")) == "D1 23:59:50–D2 00:00:10", "时间显示：有秒才显示秒")
    ok(fmt_gap(3) == "3 分钟" and fmt_gap(Fraction(4, 60)) == "4 秒" and fmt_gap(Fraction(200, 60)) == "3 分 20 秒", "时间间隔显示")
    g, sc = _good()
    _apply(g, sc, [("set", ("events", "e1", "at"), "D1 6pm")])
    ok(any("D2 02:00:43" in x.text for x in run_checks(g, scripts=sc)[1].issues if x.code == "bad_time"),
       "时间格式不对的提示里说明了可以带秒")
    g, sc = _good()
    _apply(g, sc, next(ops for n, ops, _w, _e in _CASES if n.startswith("路：带秒的间隔太短")))
    ok(any("中间只有 4 秒" in x.text for x in run_checks(g, scripts=sc)[1].issues if x.code == "walk_short"),
       "路：带秒的间隔按“4 秒”显示，不是分数")
    ok(pos_cmp((1, 5), (1, 7)) == -1 and pos_cmp((2, 1), (1, 9)) == 1 and pos_cmp((1, 3), (1, 3)) == 0
       and pos_cmp((1, None), (1, 3)) is None and pos_cmp((1, None), (2, 3)) == -1, "（集、段）比先后")
    ok(fmt_ranges([1, 2, 3, 5, 6, 9]) == "1–3、5–6、9", "段号区间显示")
    net = Net([{"from": "a", "to": "b", "walk": 1}, {"from": "b", "to": "c", "walk": 2}, {"from": "a", "to": "c", "walk": 5},
               {"from": "a", "to": "b", "walk": 0.5}, {"from": "x", "to": "y"}])
    ok(net.shortest("a", "c") == (2.5, ["a", "b", "c"]) and net.shortest("c", "a") == (2.5, ["c", "b", "a"]),
       "最短路：取最短、不分方向、重复的 links 取最快的")
    ok(net.shortest("a", "y") is None and net.shortest("x", "y") == (1, ["x", "y"]), "最短路：连不通返回空，walk 不写当 1")
    sample_script = ("# 标题\n**第 77 段 · 正文之前，不算**\n## 一句话\n无\n## 正文\n"
                     "**第 1 段 · 前台 · 傍晚（约 10 秒）**\n- 镜头\n**第 2 段 · 走廊**【新增】\n"
                     "【改】**第 3 段 · 天台**\n**第三段** 不是数字，不算\n## 附\n**第 99 段 · 正文之后，不算**\n")
    ok(parse_script(sample_script) == ([1, 2, 3], None), "认段标题：前后带【新增】【改】的也认；正文之外的不算")
    ok(parse_script("# 没有正文")[1] is not None and parse_script("## 正文\n没有段标题")[1] is not None,
       "认段标题：没有正文、没有段标题时说明原因")

    # ---- 好样本：不能误报
    g, sc = _good()
    c, r = run_checks(g, scripts=sc)
    ok(not r.issues, "好样本不该报任何问题", brief(r, set(CODES)))
    print(f"  好样本：人物 {len(c.coll['characters'])} · 事件 {len(c.coll['events'])} · 线索 {len(c.cls)}，问题 {len(r.issues)} 条")

    # ---- 每种错单独放进去，看查不查得出来、有没有乱报
    covered: set[str] = set()
    for name, ops, want, extra in _CASES:
        g, sc = _good()
        _apply(g, sc, ops)
        c, r = run_checks(g, scripts=sc)
        got = r.codes()
        covered |= want
        good_a = ok(want <= got, f"{name}：应该查出 {sorted(want)}，实际 {sorted(got)}")
        good_b = ok(got <= want | extra, f"{name}：多报了 {sorted(got - want - extra)}", brief(r, got - want - extra))
        if "只提醒一次" in name:
            code = next(iter(want))
            good_c = ok(r.count(code) == 1, f"{name}：应该只提醒 1 次，实际 {r.count(code)} 次")
        else:
            good_c = True
        mark = "✓" if good_a and good_b and good_c else "✗"
        what = "、".join(CODES[k][1] for k in sorted(want, key=list(CODES).index)) or "不报"
        print(f"  {mark} {name} → {what}")

    # ---- 一份放了很多种错的坏样本
    g, sc = _good()
    picked = {n: (o, w) for n, o, w, _e in _CASES}
    want_all: set[str] = set()
    for n in _BAD_PICK:
        _apply(g, sc, picked[n][0])
        want_all |= picked[n][1]
    c, r = run_checks(g, scripts=sc)
    ok(want_all <= r.codes(), "混合坏样本：每一类都要查出来", f"漏了 {sorted(want_all - r.codes())}")
    text = render(c, r)
    ok("✗ 必改（" in text and "⚠ 提醒（" in text and "✓ 小计：" in text and "各条故事线" in text, "输出：三级和小计都有")
    print(f"  ✓ 混合坏样本：{len(_BAD_PICK)} 处错放在一起 → 必改 {len(r.by_level(MUST))} 条、提醒 {len(r.by_level(WARN))} 条，"
          f"{len(want_all)} 类都查出来了")

    # ---- 每种问题都要有样本覆盖到（只有“字段类型不对”由下面的乱写样本覆盖）
    missing_cover = set(CODES) - covered - {"bad_type"}
    ok(not missing_cover, "有些问题种类没有自测样本", f"{sorted(missing_cover)}")

    # ---- 乱写的图谱：可以报一堆必改，但脚本自己不能崩
    garbage: dict[str, Any] = {
        "version": 1, "title": 5, "characters": "oops",
        "relations": [None, {"from": 1, "to": ["x"], "label": 3}],
        "maps": [{"id": "m", "layers": 3}, {"id": "m2", "layers": [1, {"id": 2}]}],
        "locations": [{"id": "l", "name": "L", "map": 4, "layer": []}, {"id": ["bad"], "name": 1}],
        "links": [{"from": "l", "to": "l", "walk": "fast", "see": "yes"}, "oops", {"from": {}, "to": []}],
        "arcs": [{"id": "a", "name": "A", "culprit": "kim", "suspects": 5, "type": ["x"]},
                 {"id": "b", "name": "B", "culprit": [None, 3], "suspects": [1, {"who": 2, "clearedAt": 7}]}],
        "episodes": [{"n": "1"}, {"n": 2, "beats": [{"arc": "a", "text": "t", "seg": "x", "kind": 3, "spoiler": "yes"}, 5]},
                     {"n": 2, "beats": "oops"}, {"n": True}],
        "clues": [{"id": "c", "name": "n", "arc": "a", "planted": "x", "payoff": {"ep": "1"}, "debunked": {"seg": 2},
                   "real": "yes", "pointsTo": 3},
                  {"id": "c2", "name": {}, "arc": 5, "planted": {"ep": 1, "seg": "z"}}],
        "events": [1, {"id": 5, "at": 7, "who": "ann", "where": ["x"]},
                   {"id": "ok", "at": "D1 10:00", "until": 5, "who": [None, 3, "ann"], "where": "nowhere", "what": "x",
                    "perceives": "no", "learns": [1, {"who": 4}], "ep": "1", "seg": "2", "spoiler": "maybe"},
                   {"id": "ok2", "at": "D1 10:05", "who": ["ann"], "where": "l", "what": "y",
                    "perceives": [{"who": "ann", "how": ["x"]}, {"how": "看见"}]}],
    }
    try:
        c, r = run_checks(garbage, scripts={})
        render(c, r)
        ok(r.by_level(MUST) and "bad_type" in r.codes(), "乱写的图谱：应该报必改和“字段类型不对”")
        print(f"  ✓ 乱写的图谱 → 没有崩，报了 {len(r.issues)} 条（含字段类型不对 {r.count('bad_type')} 条）")
    except Exception as ex:  # noqa: BLE001
        ok(False, "乱写的图谱不该让脚本崩掉", f"{type(ex).__name__}: {ex}")

    # ---- 读文件夹：正常用法（图谱 + EPxx/剧本.md），以及退出码
    def script_text(n: int, marks: bool) -> str:
        segs = "\n".join(f"{'【新增】' if marks and i == 2 else ''}**第 {i} 段 · 地点 · 夜（约 10 秒）**"
                         f"{'【改】' if marks and i == 3 else ''}\n- 镜头 1：……\n" for i in range(1, n + 1))
        return f"# 测试 · 第 1 集\n\n## 一句话\n不进正文\n\n## 正文\n\n{segs}\n## 附\n**第 99 段 · 不算**\n"

    def run_dir(show: Path) -> tuple[str, int]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main([str(show)])
        return buf.getvalue(), code

    with tempfile.TemporaryDirectory() as d:
        show = Path(d) / "剧"
        for ep, n in ((1, 10), (2, 5)):
            (show / f"EP{ep:02d}").mkdir(parents=True)
            (show / f"EP{ep:02d}" / "剧本.md").write_text(script_text(n, marks=True), encoding="utf-8")
        g, _ = _good()
        (show / "故事图谱.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        out, code = run_dir(show)
        ok(code == 0 and "✗ 必改（0）：没有" in out and "⚠ 提醒（0）：没有" in out and "故事图谱检查 · 自测好样本" in out,
           "文件夹：好图谱 + 剧本 → 退出码 0，没有问题", out)
        (show / "EP01" / "剧本.md").write_text(script_text(7, marks=False), encoding="utf-8")  # 剧本少了 8–10 段
        out, code = run_dir(show)
        ok(code == 1 and "里没有这一段" in out and "第 1–7 段" in out, "文件夹：剧本少了几段 → 退出码 1，说明剧本里有哪几段", out)
        shutil_target = show / "EP02"
        (shutil_target / "剧本.md").unlink()
        shutil_target.rmdir()
        out, code = run_dir(show)
        ok(out.count("第 2 集没有剧本文件") == 1, "文件夹：缺一集剧本 → 只提醒一次", out)
        (show / "故事图谱.json").write_text("{ 不是 json", encoding="utf-8")
        out, code = run_dir(show)
        ok(code == 1 and "不是合法的 JSON" in out, "文件夹：JSON 写坏了 → 退出码 1 并说明在哪", out)
        (show / "故事图谱.json").write_text("[1, 2]", encoding="utf-8")
        out, code = run_dir(show)
        ok(code == 1 and "最外层应该是一个对象" in out, "文件夹：最外层不是对象 → 退出码 1", out)
        (show / "故事图谱.json").unlink()
        out, code = run_dir(show)
        ok(code == 1 and "找不到" in out, "文件夹：没有 故事图谱.json → 退出码 1", out)
        out, code = run_dir(Path(d) / "不存在")
        ok(code == 1 and "找不到剧的文件夹" in out, "文件夹：路径不对 → 退出码 1", out)

    if fails:
        print(f"\n✗ 自测没通过（{len(fails)} 项）：")
        print("\n".join(fails))
        return 1
    print(f"\n✓ 自测全部通过：共 {passed} 项检查；好样本零误报，{len(_CASES)} 种单个错误和 1 份混合坏样本都查得出来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
