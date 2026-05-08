# ================================================
# NOTIFIER.PY - Windows Desktop + Browser Sound Alerts
# Push notifications when scanner finds qualified setups
# ================================================

import os
import sys
import json
from datetime import datetime

# === WINDOWS DESKTOP NOTIFICATIONS ===

def send_desktop_notification(title, message, duration=10):
    """
    Send a Windows desktop notification (popup in corner).
    Falls back gracefully if libraries aren't available.
    """
    # Method 1: win10toast (most reliable on Windows)
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message,
            duration=duration,
            threaded=True,
        )
        return True
    except ImportError:
        pass

    # Method 2: plyer (cross-platform)
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            timeout=duration,
            app_name="Trading Scanner",
        )
        return True
    except ImportError:
        pass

    # Method 3: Windows native (PowerShell)
    if sys.platform == 'win32':
        try:
            import subprocess
            ps_script = f"""
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
            $template = @"
            <toast>
                <visual>
                    <binding template="ToastGeneric">
                        <text>{title}</text>
                        <text>{message}</text>
                    </binding>
                </visual>
                <audio src="ms-winsoundevent:Notification.Default"/>
            </toast>
"@
            $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
            $xml.LoadXml($template)
            $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Trading Scanner").Show($toast)
            """
            subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True, timeout=5
            )
            return True
        except Exception:
            pass

    # Method 4: Simple print fallback
    print(f"\n🔔 ALERT: {title}")
    print(f"   {message}\n")
    return False


def format_scan_notification(results):
    """
    Format scan results into a notification message.
    """
    if not results or not results.get('top10'):
        return None, None

    top = results['top10']
    count = len(results['all_qualified'])
    scan_time = results['scan_time']

    title = f"🎯 Scanner: {count} Setups Found ({scan_time})"

    lines = []
    for r in top[:5]:  # Top 5 in notification (keep it short)
        arrow = "🟢" if r['direction'] == 'LONG' else "🔴"
        lines.append(f"{arrow} {r['ticker']} {r['trade_type']} ({r['composite']:+d}) {r['confidence']}%")

    if count > 5:
        lines.append(f"...and {count - 5} more")

    message = "\n".join(lines)
    return title, message


# === BROWSER SOUND ALERT (Streamlit compatible) ===

def get_alert_sound_html(sound_type="success"):
    """
    Returns HTML/JS that plays a notification sound in the browser.
    Uses Web Audio API (no external files needed).
    """
    if sound_type == "success":
        # Pleasant ascending chime
        return """
        <script>
        (function() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();

                function playTone(freq, startTime, duration) {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.frequency.value = freq;
                    osc.type = 'sine';
                    gain.gain.setValueAtTime(0.3, startTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
                    osc.start(startTime);
                    osc.stop(startTime + duration);
                }

                const now = ctx.currentTime;
                playTone(523, now, 0.15);        // C5
                playTone(659, now + 0.12, 0.15);  // E5
                playTone(784, now + 0.24, 0.2);   // G5
                playTone(1047, now + 0.36, 0.3);  // C6
            } catch(e) {}
        })();
        </script>
        """

    elif sound_type == "alert":
        # Urgent double beep
        return """
        <script>
        (function() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();

                function beep(freq, startTime, duration) {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.frequency.value = freq;
                    osc.type = 'square';
                    gain.gain.setValueAtTime(0.2, startTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
                    osc.start(startTime);
                    osc.stop(startTime + duration);
                }

                const now = ctx.currentTime;
                beep(880, now, 0.1);
                beep(880, now + 0.15, 0.1);
                beep(1100, now + 0.35, 0.15);
            } catch(e) {}
        })();
        </script>
        """

    elif sound_type == "scan_start":
        # Soft single tone
        return """
        <script>
        (function() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = 440;
                osc.type = 'sine';
                gain.gain.setValueAtTime(0.15, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.3);
            } catch(e) {}
        })();
        </script>
        """

    return ""


# === SCAN SCHEDULE CHECKER ===

SCAN_TIMES = [
    {"label": "Pre-Market", "hour": 9, "minute": 0},     # 9:00 AM ET
    {"label": "Market Open", "hour": 9, "minute": 30},    # 9:30 AM ET
    {"label": "Midday", "hour": 11, "minute": 45},        # 11:45 AM ET
    {"label": "Power Hour", "hour": 14, "minute": 45},    # 2:45 PM ET
]


def get_next_scan_time():
    """Returns the next scheduled scan time and its label."""
    now = datetime.now()
    for scan in SCAN_TIMES:
        scan_dt = now.replace(hour=scan['hour'], minute=scan['minute'], second=0, microsecond=0)
        if now < scan_dt:
            delta = scan_dt - now
            mins = int(delta.total_seconds() / 60)
            return {
                'label': scan['label'],
                'time': scan_dt.strftime("%I:%M %p"),
                'minutes_until': mins,
                'datetime': scan_dt,
            }
    return {'label': 'Tomorrow Pre-Market', 'time': '9:00 AM', 'minutes_until': None, 'datetime': None}


def should_auto_scan():
    """Check if current time matches a scan window (within 2 min)."""
    now = datetime.now()
    for scan in SCAN_TIMES:
        scan_dt = now.replace(hour=scan['hour'], minute=scan['minute'], second=0, microsecond=0)
        diff = abs((now - scan_dt).total_seconds())
        if diff <= 120:  # Within 2 minutes
            return True, scan['label']
    return False, None
