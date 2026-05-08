"""
alerts.py — Alert Management System
- Stores alerts to JSON, auto-expires after 72 hours
- Banner state management (77-second display)
- Previous alerts log
"""

import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ALERTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alerts_log.json")
ALERT_TTL_HOURS = 72


# ── Load / Save ────────────────────────────────────────────────────────────────

def load_alerts() -> list[dict]:
    """Load alerts from disk, auto-purging expired ones."""
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)
        # Purge expired
        cutoff = datetime.now(ET) - timedelta(hours=ALERT_TTL_HOURS)
        alerts = [
            a for a in alerts
            if datetime.fromisoformat(a["timestamp"]).replace(tzinfo=ET) > cutoff
        ]
        save_alerts(alerts)
        return alerts
    except Exception:
        return []


def save_alerts(alerts: list[dict]):
    try:
        with open(ALERTS_FILE, "w") as f:
            json.dump(alerts, f, indent=2, default=str)
    except Exception:
        pass


# ── Create alert ───────────────────────────────────────────────────────────────

def create_alert(
    ticker: str,
    grade: str,
    direction: str,
    regime: str,
    price: float,
    trade_type: str,
    signal_score: int,
    source: str = "scanner",  # "scanner" or "single_ticker"
) -> dict:
    """
    Create and persist a new alert.
    Returns the alert dict.
    """
    alerts = load_alerts()

    # Deduplicate — don't fire same ticker alert more than once per 30 minutes
    now = datetime.now(ET)
    for a in alerts:
        if a["ticker"] == ticker:
            last = datetime.fromisoformat(a["timestamp"]).replace(tzinfo=ET)
            if (now - last).total_seconds() < 1800:  # 30 min cooldown
                return a  # Return existing alert, don't duplicate

    alert = {
        "id":           f"{ticker}_{int(time.time())}",
        "ticker":       ticker,
        "grade":        grade,
        "direction":    direction,
        "regime":       regime,
        "price":        price,
        "trade_type":   trade_type,
        "signal_score": signal_score,
        "source":       source,
        "timestamp":    now.isoformat(),
        "acknowledged": False,
        "expired_at":   (now + timedelta(hours=ALERT_TTL_HOURS)).isoformat(),
        "banner_shown": False,
    }

    alerts.insert(0, alert)
    save_alerts(alerts)
    return alert


def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged."""
    alerts = load_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            a["acknowledged"] = True
            break
    save_alerts(alerts)


def mark_banner_shown(alert_id: str):
    """Mark that the banner was displayed for this alert."""
    alerts = load_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            a["banner_shown"] = True
            break
    save_alerts(alerts)


def get_pending_banner_alerts() -> list[dict]:
    """Get alerts that haven't had their banner shown yet."""
    alerts = load_alerts()
    return [a for a in alerts if not a.get("banner_shown") and not a.get("acknowledged")]


def clear_all_alerts():
    """Clear all alerts."""
    save_alerts([])


def get_time_remaining(alert: dict) -> str:
    """Get human-readable time until alert expires."""
    try:
        expired_at = datetime.fromisoformat(alert["expired_at"]).replace(tzinfo=ET)
        remaining  = expired_at - datetime.now(ET)
        if remaining.total_seconds() <= 0:
            return "Expired"
        hours = int(remaining.total_seconds() // 3600)
        mins  = int((remaining.total_seconds() % 3600) // 60)
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"
    except Exception:
        return "—"


# ── Banner HTML ────────────────────────────────────────────────────────────────

def get_alert_banner_html(alert: dict, banner_duration_sec: int = 77) -> str:
    """
    Returns HTML/JS for the animated 77-second alert banner.
    Shows only when user is on the dashboard.
    """
    grade   = alert.get("grade", "A")
    ticker  = alert.get("ticker", "")
    direction = alert.get("direction", "LONG")
    regime  = alert.get("regime", "")
    price   = alert.get("price", 0)
    tt      = alert.get("trade_type", "")
    score   = alert.get("signal_score", 0)
    alert_id = alert.get("id", "")

    dir_color = "#4ade80" if direction == "LONG" else "#f87171"
    dir_arrow = "▲" if direction == "LONG" else "▼"
    grade_colors = {
        "A+": ("#166534", "#4ade80", "#22c55e"),
        "A":  ("#1e3a5f", "#93c5fd", "#3b82f6"),
        "B":  ("#3b3216", "#fcd34d", "#f59e0b"),
    }
    bg, fg, border = grade_colors.get(grade, ("#1a1a2e", "#94a3b8", "#6366f1"))

    return f"""
    <div id="alert_banner_{alert_id}" style="
        position: fixed; top: 80px; right: 24px; z-index: 9999;
        width: 360px;
        background: linear-gradient(135deg, #0a0b0f, #13151f);
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 0 40px rgba(99,102,241,0.2), 0 8px 32px rgba(0,0,0,0.6);
        animation: slideIn 0.4s ease-out, pulseGlow 2s ease-in-out infinite;
        font-family: 'DM Mono', monospace;
    ">
        <style>
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateX(100px); }}
                to   {{ opacity: 1; transform: translateX(0); }}
            }}
            @keyframes pulseGlow {{
                0%, 100% {{ box-shadow: 0 0 40px rgba(99,102,241,0.2), 0 8px 32px rgba(0,0,0,0.6); }}
                50%       {{ box-shadow: 0 0 60px rgba(99,102,241,0.35), 0 8px 32px rgba(0,0,0,0.6); }}
            }}
            #timer_bar_{alert_id} {{
                height: 3px;
                background: {border};
                border-radius: 2px;
                animation: shrink {banner_duration_sec}s linear forwards;
            }}
            @keyframes shrink {{
                from {{ width: 100%; }}
                to   {{ width: 0%; }}
            }}
        </style>

        <!-- Header -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="
                    display:inline-block; padding:3px 12px; border-radius:100px;
                    background:{bg}; color:{fg}; border:1px solid {border};
                    font-size:13px; font-weight:600;
                ">{grade}</span>
                <span style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:1.5px;">
                    NEW SIGNAL
                </span>
            </div>
            <span style="font-size:11px;color:#374151;">
                {datetime.now(ET).strftime('%H:%M:%S')}
            </span>
        </div>

        <!-- Main content -->
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <span style="font-size:28px;font-weight:700;color:#e2e8f0;">{ticker}</span>
            <span style="font-size:22px;color:{dir_color};">{dir_arrow}</span>
            <div>
                <div style="font-size:14px;font-weight:600;color:{dir_color};">{direction}</div>
                <div style="font-size:11px;color:#4b5563;">{tt}</div>
            </div>
            <div style="margin-left:auto;text-align:right;">
                <div style="font-size:16px;color:#e2e8f0;">${price:.2f}</div>
                <div style="font-size:11px;color:#4b5563;">Score: {score:+d}</div>
            </div>
        </div>

        <!-- Regime -->
        <div style="font-size:12px;color:#64748b;margin-bottom:14px;">
            Regime: <span style="color:#94a3b8;">{regime}</span>
        </div>

        <!-- Timer bar -->
        <div style="background:rgba(255,255,255,0.05);border-radius:2px;margin-bottom:12px;overflow:hidden;">
            <div id="timer_bar_{alert_id}"></div>
        </div>

        <!-- Dismiss note -->
        <div style="font-size:10px;color:#374151;text-align:center;font-family:'DM Sans',sans-serif;">
            Auto-dismisses in {banner_duration_sec}s — switch to Alerts tab to acknowledge
        </div>
    </div>

    <script>
        // Auto-hide after duration
        setTimeout(function() {{
            var el = document.getElementById('alert_banner_{alert_id}');
            if (el) {{
                el.style.transition = 'opacity 0.5s';
                el.style.opacity = '0';
                setTimeout(function() {{ if(el) el.remove(); }}, 500);
            }}
        }}, {banner_duration_sec * 1000});
    </script>
    """
