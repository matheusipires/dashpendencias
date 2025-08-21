# app.py — Painel de OS (SES/GO + ORBiS) — BACKLOG (Abertas & Pendentes)
# - KPIs no topo (Total, Atrasadas, Média dias, Média OS/mês)
# - Sidebar com Período, Atualizar e Filtros (Cliente, Tipo, Situação, Pendência, Equipamento, Req)
# - Abas: Por Cliente | Por Mês/Ano | Por Tipo de Manutenção | Por Equipamento | Pendências
# - Pendências: quantidade por tipo e duração média (dias) até hoje (se sem fechamento)
# - Campos de Pendência: início e fim (quando houver). Parser robusto BR/ISO/epoch.

import os, re, unicodedata, time, json, math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import plotly.express as px
import streamlit as st
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(
    page_title="Painel OS — Backlog (SES/GO + ORBiS)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh: 60s
st.markdown(
    "<script>setTimeout(function(){window.location.reload();}, 60000);</script>",
    unsafe_allow_html=True,
)

# ========================= TEMA / ESTILO =========================
BRAND = {
    "TEAL_DARK":  "#1B556B",
    "ORANGE":     "#E98C5F",
    "GREEN":      "#32AF9D",
    "CYAN":       "#00AFD1",
    "LIGHT_BLUE": "#83D0F5",
    "YELLOW":     "#FFE596",
    "PURPLE":     "#524E9C",
    "ROSE":       "#AF5B65",
}
px.defaults.color_discrete_sequence = [
    BRAND["TEAL_DARK"], BRAND["ORANGE"], BRAND["GREEN"], BRAND["CYAN"],
    BRAND["PURPLE"], BRAND["ROSE"], BRAND["LIGHT_BLUE"], BRAND["YELLOW"],
]

st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 0.6rem; }
h1, h2, h3 { letter-spacing: .2px; }
.kpi-card {
  border-radius: 16px; padding: 14px 16px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.00));
}
.kpi-accent { height: 6px; border-radius: 6px; margin-bottom: 10px; }
.kpi-title { font-size: .95rem; opacity: .8; margin-bottom: 6px; }
.kpi-value { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
.kpi-sub { font-size: .8rem; opacity: .75; margin-top: 4px; }
.small-cap { font-size: .85rem; opacity: .8; }
</style>
""", unsafe_allow_html=True)

def style_plot(fig, height=460, headroom_factor=1.12):
    """Acabamento consistente e sem corte no topo + rótulos legíveis."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=90, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0),
        uniformtext_minsize=8, uniformtext_mode="hide",
        separators=".,",
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", automargin=True)
    fig.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)

    # headroom para não cortar rótulos
    try:
        ymax = 0
        for tr in fig.data:
            if getattr(tr, "y", None) is not None:
                try:
                    ymax = max(ymax, max([v for v in tr.y if v is not None]))
                except Exception:
                    pass
        if ymax and ymax > 0:
            fig.update_yaxes(range=[0, ymax * headroom_factor])
    except Exception:
        pass
    return fig

def render_kpi(col, title, value, accent_color, subtitle=""):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-accent" style="background:{accent_color}"></div>
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ======= Helpers de formatação (pt-BR) =======
def fmt_num_br(x: float | int, dec: int = 0) -> str:
    try:
        s = f"{float(x):,.{dec}f}"
        return s.replace(",", "§").replace(".", ",").replace("§", ".")
    except Exception:
        return str(x)

def fmt_pct_br(p: float) -> str:
    try:
        return f"{p:.1f}%".replace(".", ",")
    except Exception:
        return "0,0%"

# ========================= CONFIG =========================
STATUS_PRIORITY_ORDER = {"FECHADA": 1, "ABERTA": 2, "PENDENTE": 3, "OUTROS": 4}
DEDUP_KEY = ["Oficina", "os"]
SITUACOES_API = "1,2"            # apenas abertas & pendentes
MAX_RETRIES = 2
RETRY_SLEEP = 1.5
STRICT_OFFICE_WHITELIST = True
PEND_EMPTY_LABEL = "(Sem pendência)"

# ========================= HTTP SESSION (Retry/Backoff) =========================
SESSION = requests.Session()
retry = Retry(
    total=3, read=3, connect=3,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# ========================= FONTES =========================
SOURCES: list[dict] = [
    {
        "label": "SES/GO",
        "api_key": st.secrets.get("SESGO_API_KEY") or os.getenv("SESGO_API_KEY"),
        "base_url": "https://sesgo.api.neovero.com",
        "endpoint": "consulta_os_sesgo",
    },
    {
        "label": "ORBiS",
        "api_key": st.secrets.get("ORBIS_API_KEY") or os.getenv("ORBIS_API_KEY"),
        "base_url": "https://orbis.api.neovero.com",
        "endpoint": "consulta_os",
    },
]
SOURCES = [s for s in SOURCES if s.get("api_key")]
EXPECTED_SOURCES = {s["label"] for s in SOURCES}

# ========================= UTILS =========================
def _safe_upper(s):
    try: return str(s).upper()
    except Exception: return str(s)

def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower())
    return s.strip("_")

def _canon_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_parenthetical(text: str) -> str:
    """Remove tudo a partir do primeiro '('."""
    return re.sub(r"\s*\(.*$", "", str(text)).strip()

def _find_col(cols: list[str], exact: list[str] = None, patterns: list[str] = None) -> str | None:
    exact = [_norm_key(x) for x in (exact or [])]
    norm_map = {_norm_key(c): c for c in cols}
    for key in exact:
        if key in norm_map: return norm_map[key]
    if patterns:
        for c in cols:
            if any(re.search(p, _norm_key(c)) for p in patterns):
                return c
    return None

def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """
    Parser robusto: epoch, ISO, BR (com e sem espaço).
    Aceita 'dd/mm/aaaa hh:mm[:ss]' e 'dd/mm/aaaahh:mm[:ss]'.
    """
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
        except Exception:
            return pd.to_datetime(s, errors="coerce")

    s_str = s.astype(str).str.strip()

    s_str = s_str.str.replace(
        r"^(\d{2}/\d{2}/\d{4})(\d{2}:\d{2}(:\d{2})?)$",
        r"\1 \2",
        regex=True
    )

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m_ms = s_str.str.fullmatch(r"\d{13}")
    if m_ms.any():
        out.loc[m_ms] = pd.to_datetime(s_str.loc[m_ms].astype("int64"), unit="ms", errors="coerce")
    m_s = s_str.str.fullmatch(r"\d{10}")
    if m_s.any():
        out.loc[m_s] = pd.to_datetime(s_str.loc[m_s].astype("int64"), unit="s", errors="coerce")

    s_iso = s_str.str.replace(r"Z$", "", regex=True)\
                 .str.replace(r"([+-]\d{2}):?(\d{2})$", r"\1\2", regex=True)
    mask_iso = s_str.str.match(r"^\d{4}-\d{2}-\d{2}")
    remain = out.isna() & mask_iso
    if remain.any():
        iso_formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z","%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",  "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",    "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",       "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]
        tmp = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in iso_formats:
            m = remain & tmp.isna()
            if not m.any(): break
            tmp.loc[m] = pd.to_datetime(s_iso[m], format=fmt, errors="coerce")
        out.loc[remain] = tmp.loc[remain]

    remain = out.isna()
    if remain.any():
        br_formats_space = ["%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M","%d/%m/%Y"]
        tmp = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in br_formats_space:
            m = remain & tmp.isna()
            if not m.any(): break
            tmp.loc[m] = pd.to_datetime(s_str[m], format=fmt, errors="coerce")
        out.loc[remain] = tmp.loc[remain]

    remain = out.isna()
    if remain.any():
        out.loc[remain] = pd.to_datetime(s_str[remain], errors="coerce", dayfirst=True)

    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out

def normalize_cols_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: _norm_key(c) for c in df.columns if c != "Fonte"}
    g = df.rename(columns=ren).copy()
    for c in ["oficina","os","situacao","requisicao","pendencia","tipomanutencao"]:
        if c in g.columns: g[c] = g[c].astype(str).str.strip()
    return g

def align_by_union(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not frames: return frames
    union = []
    for df in frames:
        for c in df.columns:
            if c not in union: union.append(c)
    return [df.reindex(columns=union) for df in frames]

# ========================= FETCH (PARALELO + RETRY) =========================
def _candidate_paramsets(label: str, data_ini: datetime, data_fim: datetime) -> list[dict]:
    di_iso = data_ini.strftime("%Y-%m-%dT00:00:00")
    df_iso = data_fim.strftime("%Y-%m-%dT23:59:59")
    di_d   = data_ini.strftime("%Y-%m-%d")
    df_d   = data_fim.strftime("%Y-%m-%d")
    return [
        {"data_abertura_inicio": di_iso, "data_abertura_fim": df_iso, "situacao_int": SITUACOES_API},
        {"data_abertura_inicio": di_d,   "data_abertura_fim": df_d,   "situacao_int": SITUACOES_API},
    ]

def _do_request(url: str, headers: dict, params: dict) -> requests.Response:
    return SESSION.get(url, params=params, headers=headers, timeout=60)

def fetch_one_source(src: dict, data_ini: datetime, data_fim: datetime):
    base = src["base_url"].rstrip("/")
    headers = {"X-API-Key": src["api_key"]}
    url = f"{base}/api/queries/execute/{src['endpoint'].lstrip('/')}"
    paramsets = _candidate_paramsets(src["label"], data_ini, data_fim)
    diagnostics: list[str] = []

    for attempt in range(1, MAX_RETRIES + 2):
        for idx, ps in enumerate(paramsets, start=1):
            try:
                r = _do_request(url, headers, ps)
                if r.status_code != 200:
                    diagnostics.append(f"[{src['label']}] Tentativa {attempt}/{MAX_RETRIES+1}, PS#{idx}: HTTP {r.status_code} — {r.text[:180]}")
                    continue
                data = r.json()
                if isinstance(data, dict) and "rows" in data: data = data["rows"]
                df = pd.DataFrame(data)
                if not df.empty:
                    df["Fonte"] = src["label"]
                    return df, True, diagnostics
                else:
                    diagnostics.append(f"[{src['label']}] Tentativa {attempt}/{MAX_RETRIES+1}, PS#{idx}: resposta vazia.")
            except Exception as e:
                diagnostics.append(f"[{src['label']}] Tentativa {attempt}/{MAX_RETRIES+1}, PS#{idx}: exceção {type(e).__name__}: {e}")
        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_SLEEP * (2 ** (attempt - 1)))
    return pd.DataFrame(), False, diagnostics

def fetch_all_complete_or_keep_previous(sources, data_ini, data_fim, debug: bool=False) -> pd.DataFrame:
    frames_by_label, ok_labels = {}, set()
    diag_by_label: dict[str, list[str]] = {}
    total = len(sources)
    prog = st.progress(0.0, text="Carregando fontes...")
    with ThreadPoolExecutor(max_workers=min(8, total)) as ex:
        fut_map = {ex.submit(fetch_one_source, src, data_ini, data_fim): src for src in sources}
        done = 0
        for fut in as_completed(fut_map):
            src = fut_map[fut]
            df, ok, diags = fut.result()
            diag_by_label[src["label"]] = diags
            if ok: frames_by_label[src["label"]] = df; ok_labels.add(src["label"])
            done += 1
            prog.progress(done/total, text=f"Carregando fontes... ({done}/{total})")
    prog.empty()

    missing = [s["label"] for s in sources if s["label"] not in ok_labels]
    if missing:
        st.error("Falha ao atualizar: faltou carregar **todas** as fontes. Sem atualização.\nFaltando: " + ", ".join(missing))
        with st.expander("🔬 Detalhes técnicos (diagnóstico de fontes)", expanded=debug):
            for label in EXPECTED_SOURCES:
                st.markdown(f"**{label}**")
                logs = diag_by_label.get(label, [])
                st.code("\n".join(logs) if logs else "_Sem logs registrados._")
        return st.session_state.get("raw_all", pd.DataFrame()).copy()

    frames = [normalize_cols_for_merge(df) for df in frames_by_label.values()]
    frames = align_by_union(frames)
    raw_all = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    st.session_state.raw_all = raw_all
    st.session_state.last_fetch_ts = time.time()
    return raw_all.copy()

# ========================= NORMALIZAÇÃO & CONSOLIDAÇÃO =========================
def _map_columns_auto(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    return {
        "Abertura": _find_col(cols, exact=["abertura","data_abertura"], patterns=[r"\babert", r"^data_.*abert"]),
        "Fechamento": _find_col(cols, exact=["fechamento","data_fechamento","conclusao","data_conclusao"], patterns=[r"\bfech", r"\bconclu", r"\bsoluc"]),
        "Oficina": _find_col(cols, exact=["oficina","unidade","setor","local","filial","unidade_executora","unidade_solicitante"], patterns=[r"\boficina\b", r"\bunidade\b", r"\bsetor\b", r"\blocal\b", r"\bfilial\b"]),
        "Situação": _find_col(cols, exact=["situacao","status","estado"], patterns=[r"\bsitu", r"\bstatus", r"\bestado"]),
        "tipomanutencao": _find_col(cols, exact=["tipomanutencao","tipo_manutencao","tipo","tipo_manutencao_os"], patterns=[r"tipo.*manuten"]),
        "Requisição": _find_col(cols, exact=["requisicao","solicitacao","requisicao_os","solicitacao_os"], patterns=[r"\brequis", r"\bsolicit"]),
        "Pendencia": _find_col(cols, exact=["pendencia","pendencia_os"], patterns=[r"\bpendenc"]),
        "os": _find_col(cols, exact=["os","numero_os","n_os","num_os","codigo_os","cod_os","id_os","numero"], patterns=[r"(^|_)os($|_)", r"numero.*os", r"num.*os", r"ordem.*serv"]),
        # Equipamento (robusto)
        "Equipamento": _find_col(
            cols,
            exact=[
                "equipamento", "descricao_equipamento", "equipamento_descricao",
                "equip_desc", "asset", "patrimonio", "patrimonio_equipamento", "bem"
            ],
            patterns=[r"\bequip", r"\bpatrim", r"\bbem\b", r"asset"]
        ),
        # Pendência (início/fim robustos)
        "PendenciaInicio": _find_col(
            cols,
            exact=[
                "inicio_pendencia", "inicio_da_pendencia",
                "data_inicio_pendencia", "data_inicio_da_pendencia",
                "dt_inicio_pendencia", "inicio_pend", "data_inicio_pend",
            ],
            patterns=[r"(^|_)inic.*pend", r"(^|_)data.*inic.*pend", r"(^|_)dt.*inic.*pend"]
        ),
        "PendenciaFim": _find_col(
            cols,
            exact=[
                "fechamento_pendencia","fechamento_da_pendencia","fim_pendencia",
                "data_fechamento_pendencia","data_fechamento_da_pendencia",
                "dt_fechamento_pendencia","fech_pend",
            ],
            patterns=[r"(^|_)fech.*pend", r"(^|_)fim.*pend", r"(^|_)encerr.*pend", r"(^|_)concl.*pend", r"(^|_)data.*fech.*pend", r"(^|_)dt.*fech.*pend"]
        ),
    }

def norm_status(s: str) -> str:
    s = _safe_upper(s)
    if ("FECHAD" in s) or ("CONCLU" in s) or ("ENCERR" in s) or ("FINALIZ" in s) or ("RESOLV" in s): return "FECHADA"
    if "ABERT"  in s:  return "ABERTA"
    if "PENDEN" in s:  return "PENDENTE"
    if "CANCELAD" in s:return "CANCELADA"
    return "OUTROS"

def _fallback_find(colnames: list[str], pattern: str) -> str | None:
    for c in colnames:
        if re.search(pattern, _norm_key(c)):
            return c
    return None

def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df = df.copy(); df.columns = df.columns.str.strip()
    out_list = []

    CLEAN_NULLS = {"None": "", "none": "", "NULL": "", "null": "", "NaN": "", "nan": ""}

    for _, sub in df.groupby("Fonte", dropna=False):
        sub = sub.copy()
        colmap = _map_columns_auto(sub)

        sub["Abertura"]   = _parse_mixed_datetime(sub[colmap["Abertura"]])   if colmap["Abertura"]   else pd.NaT
        sub["Fechamento"] = _parse_mixed_datetime(sub[colmap["Fechamento"]]) if colmap["Fechamento"] else pd.NaT

        def pick(col, default=""):
            return sub[col] if col in sub.columns else pd.Series([default]*len(sub), index=sub.index)

        sub["Oficina"]           = pick(colmap["Oficina"], "NÃO INFORMADA").fillna("NÃO INFORMADA").astype(str)
        sub["os"]                = pick(colmap["os"], "").fillna("").astype(str).str.strip()
        sub["Situação_raw"]      = pick(colmap["Situação"], "").astype(str)
        sub["Situação"]          = sub["Situação_raw"].apply(norm_status)
        sub["tipomanutencao"]    = pick(colmap["tipomanutencao"], "NÃO INFORMADA").astype(str).str.upper().replace("", "NÃO INFORMADA")
        sub["Requisição"]        = pick(colmap["Requisição"], "").astype(str).str.strip()
        sub["Pendencia"]         = pick(colmap["Pendencia"], "").astype(str)

        # Equipamento — remove conteúdo entre parênteses e normaliza
        if colmap.get("Equipamento"):
            s_eq = pick(colmap["Equipamento"], "").astype(str)
            s_eq = s_eq.map(_strip_parenthetical)          # <<< remove "(...)"
            sub["Equipamento"] = s_eq.map(_canon_text)     # maiúsculas, acentos/esp.
        else:
            sub["Equipamento"] = ""

        # Datas da pendência
        pend_ini_raw = colmap.get("PendenciaInicio") or _fallback_find(sub.columns.tolist(), r"inic.*pend")
        pend_fim_raw = colmap.get("PendenciaFim")    or _fallback_find(sub.columns.tolist(), r"(fech|fim|encerr|concl).*pend")

        if pend_ini_raw:
            s_ini = sub[pend_ini_raw].astype(str).str.strip().replace(CLEAN_NULLS)
            sub["Pend_ini"] = _parse_mixed_datetime(s_ini)
        else:
            sub["Pend_ini"] = pd.NaT

        if pend_fim_raw:
            s_fim = sub[pend_fim_raw].astype(str).str.strip().replace(CLEAN_NULLS)
            sub["Pend_fim"] = _parse_mixed_datetime(s_fim)
        else:
            sub["Pend_fim"] = pd.NaT

        sub = sub[sub["Situação"] != "CANCELADA"].copy()

        empty_os = sub["os"].eq("")
        if empty_os.any():
            sub.loc[empty_os, "os"] = sub.loc[empty_os, "Requisição"].fillna("").astype(str).str.strip()
        still_empty = sub["os"].eq("")
        if still_empty.any():
            sub.loc[still_empty, "os"] = "ID_" + sub.index.astype(str)

        sub["Oficina"] = sub["Oficina"].replace("", "NÃO INFORMADA")
        out_list.append(sub)

    df2 = pd.concat(out_list, ignore_index=True, sort=False)
    df2 = df2[(df2["os"] != "") & (df2["Oficina"] != "")]
    return df2

def build_os_canonical(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()

    def status_final(series: pd.Series) -> str:
        vals = series.astype(str).map(norm_status)
        order = vals.map(STATUS_PRIORITY_ORDER).fillna(999)
        idx = order.idxmin() if len(order) else None
        return vals.loc[idx] if idx is not None else "OUTROS"

    def tipo_final(series: pd.Series) -> str:
        m = series.mode(dropna=True); return m.iat[0] if not m.empty else "NÃO INFORMADA"

    def equip_final(series: pd.Series) -> str:
        s = series.astype(str).str.strip()
        s = s[s != ""]
        if s.empty: return "NÃO INFORMADO"
        m = s.mode(dropna=True)
        return m.iat[0] if not m.empty else s.iloc[-1]

    def req_flag(series: pd.Series) -> str:
        return "SIM" if (series.astype(str).str.strip() != "").any() else ""

    def pendencia_final(series: pd.Series) -> str:
        s = series.astype(str).str.strip(); s = s[s != ""]
        if s.empty: return ""
        m = s.mode(dropna=True); return m.iat[0] if not m.empty else s.iloc[-1]

    def fontes_join(series: pd.Series) -> str:
        vals = sorted({str(x) for x in series.dropna().astype(str)})
        return ", ".join(vals) if vals else ""

    aggs = (
        df.groupby(DEDUP_KEY, dropna=False)
          .agg(
              Abertura_OS=("Abertura", "min"),
              Fechamento_OS=("Fechamento", "max"),
              Status_final=("Situação", status_final),
              Tipo_final=("tipomanutencao", tipo_final),
              Equipamento=("Equipamento", equip_final),
              Req_flag=("Requisição", req_flag),
              Pendencia_final=("Pendencia", pendencia_final),
              Fontes=("Fonte", fontes_join),
              Pendencia_Inicio=("Pend_ini", "min"),
              Pendencia_Fim=("Pend_fim", "max"),
          )
          .reset_index()
    )

    aggs["mes_abertura"] = aggs["Abertura_OS"].dt.to_period("M")
    aggs["mes_fech"]     = aggs["Fechamento_OS"].dt.to_period("M")
    aggs["fechada_mesmo_mes"] = (
        (aggs["Status_final"] == "FECHADA") &
        (aggs["Fechamento_OS"].notna()) &
        (aggs["mes_abertura"] == aggs["mes_fech"])
    )
    return aggs

# ========================= WHITELIST & MAPA (secrets) =========================
def load_office_whitelist_from_secrets() -> set[str]:
    raw = (st.secrets.get("OFICINAS_ATIVAS_JSON") if hasattr(st, "secrets") else None) or os.getenv("OFICINAS_ATIVAS_JSON", "")
    if not raw: return set()
    try:
        data = json.loads(raw)
        return {_canon_name(x) for x in data if str(x).strip()}
    except Exception as e:
        st.error(f"OFICINAS_ATIVAS_JSON inválido no secrets/ENV: {e}.")
        return set()

def load_mapping_from_secrets() -> pd.DataFrame | None:
    js = (st.secrets.get("CLIENTE_OFICINAS_JSON") if hasattr(st, "secrets") else None) or os.getenv("CLIENTE_OFICINAS_JSON", "")
    if not js: return None
    try:
        data = json.loads(js)
        df = pd.DataFrame(data).rename(columns=lambda c: c.upper())
        if {"OFICINA","CLIENTE"}.issubset(df.columns):
            return df[["OFICINA","CLIENTE"]].dropna(how="any")
    except Exception as e:
        st.error(f"CLIENTE_OFICINAS_JSON inválido: {e}")
    return None

def apply_office_whitelist_raw(raw: pd.DataFrame, whitelist: set[str], strict: bool = True) -> pd.DataFrame:
    if raw.empty: return raw.copy()
    if not whitelist: return raw.iloc[0:0] if strict else raw.copy()
    key = raw["Oficina"].map(_canon_name)
    return raw[key.isin(whitelist)].copy()

def apply_office_to_client(can: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    m = map_df.copy().rename(columns=lambda c: c.upper())
    m = m[["OFICINA","CLIENTE"]].dropna(how="any")
    m["OFI_CANON"] = m["OFICINA"].map(_canon_name)
    m = m.drop_duplicates(subset=["OFI_CANON"], keep="first")

    exact_map = dict(zip(m["OFI_CANON"], m["CLIENTE"]))

    out = can.copy()
    out["OFI_CANON"] = out["Oficina"].map(_canon_name)
    out["Cliente"]   = out["OFI_CANON"].map(exact_map)
    out = out[out["Cliente"].notna()].copy()

    out = (out.sort_values(["Cliente","os","Abertura_OS"])
              .drop_duplicates(subset=["Cliente","os"], keep="first"))
    out.drop(columns=["OFI_CANON"], inplace=True, errors="ignore")
    return out

# ========================= SIDEBAR CONTROLS =========================
def sidebar_controls():
    st.sidebar.markdown("## 🧭 Painel de OS — Backlog")
    st.sidebar.caption("Foco em OS **Abertas** ou com **Pendência**. Autoatualização a cada 1 min.")

    st.sidebar.markdown("### 📅 Período (Abertura)")
    data_ini = st.sidebar.date_input("De", value=datetime.today() - timedelta(days=30), key="dt_ini")
    data_fim = st.sidebar.date_input("Até", value=datetime.today(), key="dt_fim")

    reload_clicked = st.sidebar.button("🔄 Atualizar dados das fontes", use_container_width=True)

    try:
        qp_debug = str(getattr(st, "query_params", {}).get("debug", "0")).lower()
    except Exception:
        qp_debug = str(st.experimental_get_query_params().get("debug", ["0"])[0]).lower()
    param_debug = qp_debug in ("1", "true", "yes")
    debug_ui = st.sidebar.checkbox("🔬 Detalhes técnicos", value=param_debug)
    DEBUG_MODE = bool(debug_ui)

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filtros")

    return data_ini, data_fim, reload_clicked, DEBUG_MODE

# ========================= APP (dados base) =========================
if not SOURCES:
    st.error("Nenhuma fonte configurada. Defina SESGO_API_KEY e/ou ORBIS_API_KEY nos secrets/ENV.")
    st.stop()

office_whitelist = load_office_whitelist_from_secrets()
map_df = load_mapping_from_secrets()
if not office_whitelist:
    st.error("OFICINAS_ATIVAS_JSON ausente ou inválido no secrets/ENV.")
    st.stop()
if map_df is None or map_df.empty:
    st.error("CLIENTE_OFICINAS_JSON ausente ou inválido no secrets/ENV (OFICINA → CLIENTE).")
    st.stop()

data_ini, data_fim, reload_clicked, DEBUG_MODE = sidebar_controls()
st.session_state["DEBUG_MODE"] = DEBUG_MODE

# Primeiro fetch (ou quando clicar)
if reload_clicked or "raw_all" not in st.session_state:
    raw_all_new = fetch_all_complete_or_keep_previous(SOURCES, data_ini, data_fim, debug=DEBUG_MODE)
    if raw_all_new.empty and "raw_all" not in st.session_state:
        st.stop()

raw_all = st.session_state.get("raw_all", pd.DataFrame()).copy()

# Pré-processamento & whitelist
raw = preprocess_raw(raw_all)
raw = apply_office_whitelist_raw(raw, office_whitelist, strict=STRICT_OFFICE_WHITELIST)
if raw.empty:
    st.warning("Nenhum dado após aplicar a whitelist de OFICINAS. Revise OFICINAS_ATIVAS_JSON no secrets.")
    st.stop()

# Consolidação e mapeamento
can = build_os_canonical(raw)
can_cli = apply_office_to_client(can, map_df)

# Recorte por período (Abertura)
mask_periodo = (can_cli["Abertura_OS"] >= pd.to_datetime(data_ini)) & \
               (can_cli["Abertura_OS"] <= pd.to_datetime(data_fim) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))
can_cli = can_cli[mask_periodo].copy()

# ========================= FILTROS FINAIS =========================
def aplicar_filtros(can_cli: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    clientes_all = sorted(can_cli["Cliente"].dropna().astype(str).unique().tolist())
    tipos_all    = sorted(can_cli["Tipo_final"].dropna().astype(str).unique().tolist())
    equips_all   = sorted(can_cli["Equipamento"].fillna("").astype(str).replace("", "NÃO INFORMADO").unique().tolist())
    pend_all_raw = can_cli["Pendencia_final"].fillna("").astype(str).str.strip()
    pend_all = sorted([(PEND_EMPTY_LABEL if p == "" else p) for p in pend_all_raw.unique().tolist()])

    if "applied_filters" not in st.session_state:
        st.session_state.applied_filters = {
            "clientes": clientes_all[:],
            "tipos": tipos_all[:],
            "equips": equips_all[:],
            "pendencias": pend_all[:],
            "situacao": "Todas",       # "Aberta" | "Pendente" | "Todas"
            "somente_req": False,
        }

    def _multi_with_actions(label: str, options: list[str], key_prefix: str, container):
        def _uniq(seq): return list(dict.fromkeys(seq))
        options = _uniq([str(o) for o in options if str(o).strip() or o == PEND_EMPTY_LABEL])
        sel_key, opts_key, wkey = f"{key_prefix}_sel", f"{key_prefix}_opts", f"{key_prefix}_widget"
        reset_widget = False
        if st.session_state.get(opts_key) != options:
            st.session_state[opts_key] = options[:]; reset_widget = True
        stored = st.session_state.get(sel_key, options[:])
        stored = [o for o in _uniq(stored) if o in options]
        if not stored and options: stored = options[:]
        st.session_state[sel_key] = stored
        c1, c2, c3 = container.columns(3)
        with c1:
            if st.button("Todos", key=f"{key_prefix}_all"): st.session_state[sel_key] = options[:]; reset_widget = True
        with c2:
            if st.button("Limpar", key=f"{key_prefix}_none"): st.session_state[sel_key] = []; reset_widget = True
        with c3:
            if st.button("Inverter", key=f"{key_prefix}_invert"):
                cur = set(st.session_state[sel_key])
                st.session_state[sel_key] = [o for o in options if o not in cur]; reset_widget = True
        if reset_widget and wkey in st.session_state: del st.session_state[wkey]
        if not options:
            container.info("Sem opções disponíveis."); st.session_state[sel_key] = []; return []
        sel = container.multiselect(label, options=options, default=st.session_state[sel_key], key=wkey)
        st.session_state[sel_key] = [o for o in _uniq(sel) if o in options]
        return st.session_state[sel_key]

    st.sidebar.subheader("Cliente")
    q_cli = st.sidebar.text_input("🔎 Buscar Cliente", placeholder="Digite parte do nome")
    clientes_opts = [c for c in clientes_all if (q_cli.lower() in c.lower())] if q_cli else clientes_all
    clientes_sel_ui = _multi_with_actions("Clientes (ativos)", clientes_opts, "f_clientes", st.sidebar)

    st.sidebar.subheader("Tipo de Manutenção")
    tipos_sel_ui = _multi_with_actions("Tipos", tipos_all, "f_tipos", st.sidebar)

    st.sidebar.subheader("Equipamento")
    q_eqp = st.sidebar.text_input("🔎 Buscar Equipamento", placeholder="Digite parte do nome")
    equips_opts = [e for e in equips_all if (q_eqp.lower() in e.lower())] if q_eqp else equips_all
    equips_sel_ui = _multi_with_actions("Equipamentos", equips_opts, "f_equips", st.sidebar)

    st.sidebar.subheader("Situação")
    situacao_ui = st.sidebar.radio(
        "Exibir", ["Aberta", "Pendente", "Todas"],
        index=["Aberta","Pendente","Todas"].index(st.session_state.applied_filters.get("situacao","Todas")),
        horizontal=True, key="f_situacao_mode"
    )

    st.sidebar.subheader("Pendência")
    pend_sel_ui = _multi_with_actions("Tipos de pendência", pend_all, "f_pend", st.sidebar)

    st.sidebar.subheader("Outros")
    somente_req_ui = st.sidebar.checkbox(
        "Somente com Requisição preenchida",
        value=bool(st.session_state.applied_filters.get("somente_req", False)),
        key="f_req_only"
    )

    apply_clicked = st.sidebar.button("✅ Aplicar filtros", use_container_width=True)
    if apply_clicked:
        st.session_state.applied_filters = {
            "clientes": [c for c in clientes_sel_ui if c in clientes_all],
            "tipos":    [t for t in tipos_sel_ui if t in tipos_all],
            "equips":   [e for e in equips_sel_ui if e in equips_all],
            "pendencias": [p for p in pend_sel_ui if p in pend_all],
            "situacao": situacao_ui,
            "somente_req": bool(somente_req_ui),
        }
        st.rerun()

    af = st.session_state.applied_filters
    out = can_cli.copy()

    out = out[out["Status_final"] != "FECHADA"]

    lab = out["Pendencia_final"].fillna("").astype(str).str.strip()
    out["Categoria"] = lab.where(lab != "", other="ABERTA")
    out["Categoria"] = out["Categoria"].apply(lambda x: "PENDENTE" if x != "ABERTA" else "ABERTA")

    if af.get("situacao") == "Aberta":
        out = out[out["Categoria"] == "ABERTA"]
    elif af.get("situacao") == "Pendente":
        out = out[out["Categoria"] == "PENDENTE"]

    out["PendTag"] = out["Pendencia_final"].apply(lambda x: PEND_EMPTY_LABEL if (str(x).strip() == "") else str(x))
    sel_pend = [p for p in af.get("pendencias", []) if p in pend_all]
    out = out[out["PendTag"].isin(sel_pend)] if sel_pend else out.iloc[0:0]

    sel_clientes = [c for c in af.get("clientes", []) if c in clientes_all]
    sel_tipos    = [t for t in af.get("tipos", [])    if t in tipos_all]
    sel_equips   = [e for e in af.get("equips", [])   if e in equips_all]

    out = out[out["Cliente"].isin(sel_clientes)] if sel_clientes else out.iloc[0:0]
    out = out[out["Tipo_final"].isin(sel_tipos)] if sel_tipos else out.iloc[0:0]
    out = out[out["Equipamento"].replace("", "NÃO INFORMADO").isin(sel_equips)] if sel_equips else out.iloc[0:0]

    if af.get("somente_req", False):
        out = out[out["Req_flag"] == "SIM"]

    st.sidebar.caption(
        f"Aplicados — Clientes: {len(sel_clientes)} • Tipos: {len(sel_tipos)} • "
        f"Equip.: {len(sel_equips)} • Situação: {af.get('situacao')} • "
        f"Pendência: {len(sel_pend)} • Req: {'Sim' if af.get('somente_req', False) else 'Não'}\n\n"
        f"OS após filtros: {len(out):,}"
    )
    return out, af

can_filt, filtros_info = aplicar_filtros(can_cli)

# ========================= MÉTRICAS (cards) =========================
def compute_metrics(df: pd.DataFrame, period_ini: datetime, period_fim: datetime):
    now = datetime.now()
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_os = df["os"].nunique() if not df.empty else 0

    atrasadas = 0
    media_dias = 0.0
    media_por_mes = 0.0

    if not df.empty:
        atrasadas = df.loc[df["Abertura_OS"] < pd.to_datetime(first_of_month), "os"].nunique()
        delta = (pd.Timestamp(now) - df["Abertura_OS"]).dt.total_seconds() / (3600*24)
        media_dias = float(delta.mean()) if not math.isnan(delta.mean()) else 0.0
        df2 = df.copy()
        df2["mes_period"] = df2["Abertura_OS"].dt.to_period("M")
        counts = df2.groupby("mes_period")["os"].nunique()
        full_range = pd.period_range(pd.to_datetime(period_ini).to_period("M"),
                                     pd.to_datetime(period_fim).to_period("M"), freq="M")
        counts = counts.reindex(full_range, fill_value=0)
        media_por_mes = float(counts.mean()) if len(counts) > 0 else 0.0

    return dict(
        total_os=int(total_os),
        atrasadas=int(atrasadas),
        media_dias=round(media_dias, 1),
        media_por_mes=round(media_por_mes, 1),
    )

metrics = compute_metrics(can_filt, data_ini, data_fim)
c1, c2, c3, c4 = st.columns(4)
render_kpi(c1, "Total de OS", fmt_num_br(metrics['total_os']), BRAND["TEAL_DARK"], "Após filtros aplicados")
render_kpi(c2, "Total atrasadas", fmt_num_br(metrics['atrasadas']), BRAND["ORANGE"], "Abertas antes do mês atual")
render_kpi(c3, "Média tempo (dias)", fmt_num_br(metrics['media_dias'], 1), BRAND["GREEN"], "OS abertas/pendentes")
render_kpi(c4, "Média de OS/mês", fmt_num_br(metrics['media_por_mes'], 1), BRAND["CYAN"], f"Período: {data_ini.strftime('%m/%Y')}–{data_fim.strftime('%m/%Y')}")

last_ts = st.session_state.get("last_fetch_ts")
if last_ts:
    st.caption(f"Última atualização: {datetime.fromtimestamp(last_ts).strftime('%d/%m/%Y %H:%M:%S')}  •  Período: {data_ini.strftime('%d/%m/%Y')}–{data_fim.strftime('%d/%m/%Y')}")

# ========================= CONTROLES DE CLASSIFICAÇÃO =========================
def sort_controls(prefix: str, allow_top=True, default_order="desc",
                  default_top=30, max_top=100):
    cols = st.columns((1, 3)) if allow_top else st.columns(1)
    with cols[0]:
        order = st.radio(
            "Ordem",
            ["Decrescente", "Crescente"],
            index=0 if default_order == "desc" else 1,
            key=f"{prefix}_ord",
            horizontal=True
        )
    topn = None
    if allow_top:
        available = int(st.session_state.get(f"{prefix}_max", 0))
        top_limit = max(0, min(max_top, available))
        if top_limit <= 1:
            st.caption("Exibindo todas as categorias")
            topn = None
        else:
            with cols[1]:
                min_val = 1 if top_limit <= 5 else 5
                if min_val >= top_limit:
                    st.caption("Exibindo todas as categorias")
                    topn = None
                else:
                    default_val = max(min_val, min(default_top, top_limit))
                    topn = st.slider("Top N", min_value=min_val, max_value=top_limit,
                                     value=default_val, step=1, key=f"{prefix}_top")
    return order, topn

# ========================= HELPER: Tabela de linhas usadas =========================
def show_rows(df_rows: pd.DataFrame, add_cols: list[str] = None, label: str = "🔎 Linhas utilizadas (OS)"):
    if df_rows.empty:
        st.info("Sem linhas para exibir com os filtros atuais.")
        return
    cols_base = ["Cliente","Oficina","os","Abertura_OS","Status_final","Tipo_final","Equipamento","Pendencia_final","Pendencia_Inicio","Fontes"]
    if "Pendencia_Fim" in df_rows.columns and df_rows["Pendencia_Fim"].notna().any():
        cols_base.insert(cols_base.index("Pendencia_Inicio")+1, "Pendencia_Fim")
    if add_cols:
        for c in add_cols:
            if c not in cols_base and c in df_rows.columns:
                cols_base.append(c)
    cols_present = [c for c in cols_base if c in df_rows.columns]
    display_df = df_rows[cols_present].copy()
    for dc in ["Abertura_OS","Pendencia_Inicio","Pendencia_Fim"]:
        if dc in display_df.columns:
            display_df[dc] = pd.to_datetime(display_df[dc], errors="coerce").dt.strftime("%d/%m/%Y %H:%M:%S")
            display_df[dc] = display_df[dc].fillna("")
    with st.expander(label):
        st.dataframe(display_df.sort_values(["Cliente","Abertura_OS","os"]),
                     use_container_width=True, height=420)

# ========================= DASHBOARDS =========================
def dash_por_cliente(df: pd.DataFrame):
    st.subheader("📊 Por Cliente")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais."); return

    grp = df.groupby("Cliente")["os"].nunique().reset_index(name="Quantidade")
    total = int(grp["Quantidade"].sum())
    grp["Pct"] = grp["Quantidade"].div(total).fillna(0) * 100
    grp["QtdLabel"] = grp["Quantidade"].map(lambda v: fmt_num_br(v))

    st.session_state["cli_max"] = int(len(grp))
    order, topn = sort_controls("cli", allow_top=True, default_order="desc", default_top=30, max_top=200)
    ascending = (order == "Crescente")

    grp_sorted = grp.sort_values("Quantidade", ascending=ascending)
    if topn: grp_sorted = grp_sorted.head(topn)

    fig = px.bar(grp_sorted, x="Cliente", y="Quantidade", text="QtdLabel")
    fig.update_xaxes(tickangle=-30, categoryorder="array", categoryarray=grp_sorted["Cliente"].tolist())
    fig.update_traces(customdata=grp_sorted["Pct"].values,
                      hovertemplate="<b>%{x}</b><br>Qtd: %{y}<br>Part: %{customdata:.1f}%<extra></extra>")
    st.plotly_chart(style_plot(fig), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    with st.expander("📋 Tabela por Cliente"):
        tbl = grp_sorted[["Cliente","Quantidade","Pct"]].copy()
        tbl["Pct"] = tbl["Pct"].map(fmt_pct_br)
        st.dataframe(tbl.set_index("Cliente"), use_container_width=True, height=420)

    base_rows = df[df["Cliente"].isin(grp_sorted["Cliente"])]
    show_rows(base_rows)

def dash_por_mes(df: pd.DataFrame):
    st.subheader("🗓️ Por Mês/Ano (mês de abertura)")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais."); return

    base = df.copy()
    base["mes_period"] = base["Abertura_OS"].dt.to_period("M")
    grp = base.groupby("mes_period")["os"].nunique().reset_index(name="Quantidade")
    if grp.empty:
        st.info("Sem dados no período."); return

    idx = pd.period_range(grp["mes_period"].min(), grp["mes_period"].max(), freq="M")
    grp = grp.set_index("mes_period").reindex(idx, fill_value=0).rename_axis("mes_period").reset_index()
    grp["mes_label"] = grp["mes_period"].apply(lambda p: f"{p.month:02d}/{p.year}")

    total = int(grp["Quantidade"].sum())
    grp["Pct"] = grp["Quantidade"].div(total).fillna(0) * 100
    grp["QtdLabel"] = grp["Quantidade"].map(lambda v: fmt_num_br(v))

    mode = st.radio("Classificação", ["Cronológico", "Quantidade"], index=0, key="mes_mode", horizontal=True)
    if mode == "Quantidade":
        ordem = st.radio("Ordem", ["Decrescente", "Crescente"], index=0, key="mes_ord", horizontal=True)
        grp = grp.sort_values("Quantidade", ascending=(ordem=="Crescente"))

    fig = px.bar(grp, x="mes_label", y="Quantidade", text="QtdLabel")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=grp["mes_label"].tolist(), tickangle=-20)
    fig.update_traces(customdata=grp["Pct"].values,
                      hovertemplate="<b>%{x}</b><br>Qtd: %{y}<br>Part: %{customdata:.1f}%<extra></extra>")
    st.plotly_chart(style_plot(fig), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    with st.expander("📋 Tabela por Mês/Ano"):
        tbl = grp[["mes_period","Quantidade","Pct"]].copy()
        tbl["Pct"] = tbl["Pct"].map(fmt_pct_br)
        st.dataframe(tbl.set_index("mes_period"), use_container_width=True, height=360)

    min_p, max_p = grp["mes_period"].min(), grp["mes_period"].max()
    mask_rows = (base["mes_period"] >= min_p) & (base["mes_period"] <= max_p)
    show_rows(base.loc[mask_rows], add_cols=["mes_abertura"])

def dash_por_tipo(df: pd.DataFrame):
    st.subheader("🧰 Por Tipo de Manutenção")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais."); return

    grp = df.groupby("Tipo_final")["os"].nunique().reset_index(name="Quantidade")
    total = int(grp["Quantidade"].sum())
    grp["Pct"] = grp["Quantidade"].div(total).fillna(0) * 100
    grp["QtdLabel"] = grp["Quantidade"].map(lambda v: fmt_num_br(v))

    st.session_state["tipo_max"] = int(len(grp))
    order, topn = sort_controls("tipo", allow_top=True, default_order="desc", default_top=20, max_top=100)
    ascending = (order == "Crescente")

    grp_sorted = grp.sort_values("Quantidade", ascending=ascending)
    if topn: grp_sorted = grp_sorted.head(topn)

    fig = px.bar(grp_sorted, x="Tipo_final", y="Quantidade", text="QtdLabel")
    fig.update_xaxes(tickangle=-25, categoryorder="array", categoryarray=grp_sorted["Tipo_final"].tolist())
    fig.update_traces(customdata=grp_sorted["Pct"].values,
                      hovertemplate="<b>%{x}</b><br>Qtd: %{y}<br>Part: %{customdata:.1f}%<extra></extra>")
    st.plotly_chart(style_plot(fig), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    with st.expander("📋 Tabela por Tipo"):
        tbl = grp_sorted[["Tipo_final","Quantidade","Pct"]].copy()
        tbl["Pct"] = tbl["Pct"].map(fmt_pct_br)
        st.dataframe(tbl.set_index("Tipo_final"), use_container_width=True, height=420)

    base_rows = df[df["Tipo_final"].isin(grp_sorted["Tipo_final"])]
    show_rows(base_rows)

def dash_por_equip(df: pd.DataFrame):
    st.subheader("🛠️ Por Equipamento")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais."); return

    base = df.copy()
    base["Equipamento"] = base["Equipamento"].replace("", "NÃO INFORMADO")

    # 🔎 Busca rápida + destaque
    st.markdown("###### 🔎 Busca rápida")
    all_eq = sorted(base["Equipamento"].unique().tolist())
    q = st.text_input("Buscar equipamento", key="eqp_q", placeholder="Digite parte do nome")
    match_opts = [e for e in all_eq if (not q or q.upper() in e.upper())]
    sel = st.selectbox("Equipamento (opcional)", ["(nenhum)"] + match_opts, index=0, key="eqp_sel")
    only_sel = st.checkbox("Mostrar apenas selecionado", value=False, key="eqp_only")

    grp = base.groupby("Equipamento")["os"].nunique().reset_index(name="Quantidade")
    total = int(grp["Quantidade"].sum())
    grp["Pct"] = grp["Quantidade"].div(total).fillna(0) * 100
    grp["QtdLabel"] = grp["Quantidade"].map(lambda v: fmt_num_br(v))

    st.session_state["equip_max"] = int(len(grp))
    order, topn = sort_controls("equip", allow_top=True, default_order="desc", default_top=30, max_top=200)
    ascending = (order == "Crescente")

    grp_sorted = grp.sort_values("Quantidade", ascending=ascending)
    if topn: grp_sorted = grp_sorted.head(topn)

    selected = sel if sel and sel != "(nenhum)" else None
    if selected and only_sel:
        grp_plot = grp_sorted[grp_sorted["Equipamento"] == selected]
        color_col = None
    else:
        grp_plot = grp_sorted.copy()
        if selected:
            grp_plot["Marca"] = grp_plot["Equipamento"].apply(lambda x: "Selecionado" if x == selected else "Outros")
            color_col = "Marca"
        else:
            color_col = None

    if color_col:
        fig = px.bar(
            grp_plot, x="Equipamento", y="Quantidade", text="QtdLabel",
            color=color_col,
            color_discrete_map={"Selecionado": BRAND["ORANGE"], "Outros": BRAND["TEAL_DARK"]}
        )
    else:
        fig = px.bar(grp_plot, x="Equipamento", y="Quantidade", text="QtdLabel")

    fig.update_xaxes(tickangle=-25, categoryorder="array", categoryarray=grp_plot["Equipamento"].tolist())
    fig.update_traces(customdata=grp_plot["Pct"].values,
                      hovertemplate="<b>%{x}</b><br>Qtd: %{y}<br>Part: %{customdata:.1f}%<extra></extra>")
    st.plotly_chart(style_plot(fig), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    with st.expander("📋 Tabela por Equipamento"):
        tbl = grp_plot[["Equipamento","Quantidade","Pct"]].copy()
        tbl["Pct"] = tbl["Pct"].map(fmt_pct_br)
        st.dataframe(tbl.set_index("Equipamento"), use_container_width=True, height=420)

    base_rows = base[base["Equipamento"].isin(grp_plot["Equipamento"])]
    show_rows(base_rows)

# ========================= PENDÊNCIAS =========================
def dash_pendencias(df: pd.DataFrame):
    st.subheader("⏳ Pendências — Quantidade e Duração média (até hoje ou fechamento)")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais."); 
        return

    pend_base = df[df["Pendencia_final"].fillna("").astype(str).str.strip() != ""].copy()
    if pend_base.empty:
        st.info("Não há OS com pendência nomeada no recorte atual.")
        show_rows(df); 
        return

    st.markdown("##### 📦 Quantidade por tipo de pendência")
    grp_qtd = pend_base.groupby("Pendencia_final")["os"].nunique().reset_index(name="Quantidade")
    total_q = int(grp_qtd["Quantidade"].sum())
    grp_qtd["Pct"] = grp_qtd["Quantidade"].div(total_q).fillna(0) * 100
    grp_qtd["QtdLabel"] = grp_qtd["Quantidade"].map(lambda v: fmt_num_br(v))

    st.session_state["pend_qtd_max"] = int(len(grp_qtd))
    order_q, topn_q = sort_controls("pend_qtd", allow_top=True, default_order="desc", default_top=20, max_top=100)
    ascending_q = (order_q == "Crescente")
    grp_qtd_sorted = grp_qtd.sort_values("Quantidade", ascending=ascending_q)
    if topn_q: grp_qtd_sorted = grp_qtd_sorted.head(topn_q)

    fig1 = px.bar(grp_qtd_sorted, x="Pendencia_final", y="Quantidade", text="QtdLabel")
    fig1.update_xaxes(tickangle=-25, categoryorder="array", categoryarray=grp_qtd_sorted["Pendencia_final"].tolist())
    fig1.update_traces(customdata=grp_qtd_sorted["Pct"].values,
                       hovertemplate="<b>%{x}</b><br>Qtd: %{y}<br>Part: %{customdata:.1f}%<extra></extra>")
    st.plotly_chart(style_plot(fig1), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    with st.expander("📋 Tabela — Quantidade por pendência"):
        tbl1 = grp_qtd_sorted[["Pendencia_final","Quantidade","Pct"]].copy()
        tbl1["Pct"] = tbl1["Pct"].map(fmt_pct_br)
        st.dataframe(tbl1.set_index("Pendencia_final"), use_container_width=True, height=360)

    st.markdown("##### 📈 Duração média (dias) por pendência (até hoje ou fechamento)")
    now = pd.Timestamp(datetime.now())
    pend_base_valid = pend_base[pend_base["Pendencia_Inicio"].notna()].copy()

    if pend_base_valid.empty:
        st.info("Não há datas de início de pendência suficientes para calcular as durações.")
    else:
        pend_base_valid["Pend_end_eff"] = pend_base_valid["Pendencia_Fim"].where(
            pend_base_valid["Pendencia_Fim"].notna(), other=now
        )
        pend_base_valid["Duracao_dias"] = (
            (pend_base_valid["Pend_end_eff"] - pend_base_valid["Pendencia_Inicio"])
            .dt.total_seconds() / (3600 * 24)
        ).clip(lower=0)

        grp_dur = pend_base_valid.groupby("Pendencia_final")["Duracao_dias"].mean().reset_index()
        grp_dur["Label"] = grp_dur["Duracao_dias"].map(lambda v: f"{fmt_num_br(v,1)} dias")
        grp_dur["Duracao_dias"] = grp_dur["Duracao_dias"].round(1)

        st.session_state["pend_dur_max"] = int(len(grp_dur))
        order_d, topn_d = sort_controls("pend_dur", allow_top=True, default_order="desc", default_top=20, max_top=100)
        ascending_d = (order_d == "Crescente")
        grp_dur_sorted = grp_dur.sort_values("Duracao_dias", ascending=ascending_d)
        if topn_d:
            grp_dur_sorted = grp_dur_sorted.head(topn_d)

        fig2 = px.bar(grp_dur_sorted, x="Pendencia_final", y="Duracao_dias", text="Label")
        fig2.update_xaxes(tickangle=-25, categoryorder="array", categoryarray=grp_dur_sorted["Pendencia_final"].tolist())
        st.plotly_chart(style_plot(fig2), use_container_width=True, theme="streamlit", config={"displayModeBar": False})

        with st.expander("📋 Tabela — Duração média (dias) por pendência (até hoje ou fechamento)"):
            st.dataframe(grp_dur_sorted.set_index("Pendencia_final"), use_container_width=True, height=360)

    show_rows(pend_base, add_cols=["Pendencia_Inicio","Pendencia_Fim"])

# ========================= RENDER =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Por Cliente",
    "2) Por Mês/Ano",
    "3) Por Tipo de Manutenção",
    "4) Por Equipamento",
    "5) Pendências"
])
with tab1: dash_por_cliente(can_filt)
with tab2: dash_por_mes(can_filt)
with tab3: dash_por_tipo(can_filt)
with tab4: dash_por_equip(can_filt)
with tab5: dash_pendencias(can_filt)
