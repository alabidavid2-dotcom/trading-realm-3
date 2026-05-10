-- ============================================================
-- TRADING REALM — trade_log table migration
-- Run this once in your Supabase SQL Editor:
--   Dashboard → SQL Editor → New Query → paste → Run
-- ============================================================

CREATE TABLE IF NOT EXISTS public.trade_log (
  id                BIGSERIAL PRIMARY KEY,

  -- Contract identity
  ticker            TEXT          NOT NULL,
  occ_symbol        TEXT,
  contract_type     TEXT,                          -- 'CALL' | 'PUT'
  strike            NUMERIC(10,2),
  expiration        TEXT,                          -- ISO date string
  dte               INTEGER,

  -- Entry
  entry_premium     NUMERIC(10,4),                 -- ask price per share
  spot_price        NUMERIC(10,2),                 -- underlying at entry
  contracts         INTEGER,
  total_cost        NUMERIC(10,2),
  trade_type        TEXT,                          -- 'intraday' | 'swing'
  direction         TEXT,                          -- 'LONG' | 'SHORT'
  strategy          TEXT,                          -- 'Gap and Go', etc.
  source            TEXT,                          -- 'live' | 'estimate'

  -- Brain context at entry
  regime            TEXT,
  regime_confidence NUMERIC(5,1),
  ftfc_aligned      INTEGER,                       -- e.g. 6
  ftfc_total        INTEGER,                       -- e.g. 7
  gap_type          TEXT,
  alpha_setup       BOOLEAN       DEFAULT FALSE,
  sector_etf        TEXT,
  sentinel_bonus    INTEGER       DEFAULT 0,

  -- Runner protocol
  scale_qty         INTEGER,
  runner_qty        INTEGER,
  runner_active     BOOLEAN       DEFAULT FALSE,

  -- Exit (filled on close)
  exit_premium      NUMERIC(10,4),
  exit_reason       TEXT,
  peak_premium      NUMERIC(10,4),                 -- HOD reached before exit
  pnl_dollars       NUMERIC(10,2),
  pnl_pct           NUMERIC(8,2),

  -- Status
  status            TEXT          DEFAULT 'open',  -- 'open' | 'closed'
  paper             BOOLEAN       DEFAULT TRUE,

  -- Timestamps
  entered_at        TIMESTAMPTZ   DEFAULT NOW(),
  exited_at         TIMESTAMPTZ,
  created_at        TIMESTAMPTZ   DEFAULT NOW()
);

-- Indexes for the analytics queries
CREATE INDEX IF NOT EXISTS idx_tl_ticker     ON public.trade_log (ticker);
CREATE INDEX IF NOT EXISTS idx_tl_regime     ON public.trade_log (regime);
CREATE INDEX IF NOT EXISTS idx_tl_status     ON public.trade_log (status);
CREATE INDEX IF NOT EXISTS idx_tl_occ        ON public.trade_log (occ_symbol);
CREATE INDEX IF NOT EXISTS idx_tl_entered    ON public.trade_log (entered_at DESC);
CREATE INDEX IF NOT EXISTS idx_tl_alpha      ON public.trade_log (alpha_setup);

-- Enable Row Level Security but allow all for now (tighten per-user later)
ALTER TABLE public.trade_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "allow_all" ON public.trade_log FOR ALL USING (true) WITH CHECK (true);
