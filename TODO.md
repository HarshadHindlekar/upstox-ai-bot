# 📋 Upstox AI Bot - Future Roadmap & TODOs

This document tracks planned features, enhancements, and milestones for upcoming development phases.

---

## 🔔 Phase 1: Real-Time Mobile Notifications & Webhooks (Option 3)

### 1. Telegram Bot Integration
- [ ] **Bot Setup & Configuration:**
  - Create bot via @BotFather and configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env / Google Drive storage.
  - Add helper module core/notifier.py with asynchronous message dispatching.
- [ ] **Order & Signal Alerts:**
  - 🟢 BUY Alert: Instant ping on high-confidence breakout (>65% probability) with Entry Price, Target (TP), and Stop-Loss (SL).
  - 🔴 EXIT Alert: Instant alert when SL hit, TP reached, or emergency square-off triggered with realized P&L.
- [ ] **Risk & Safety Alerts:**
  - Immediate high-priority alert if the Daily Drawdown Kill-Switch is triggered (-₹2,000 threshold reached).
- [ ] **End-of-Day (EOD) Digest:**
  - Automated session summary sent at 3:30 PM IST with net P&L, win rate, total trades executed, and open position count.

### 2. Discord Webhook Integration
- [ ] **Channel Webhook Alerts:**
  - Add DISCORD_WEBHOOK_URL support for formatted embed cards displaying rich trade summaries, color-coded P&L (green/red), and execution timestamps.

---

## 🚀 Phase 2: Showcase, Community & Portfolio Polish (Option 4)

### 1. Repository Presentation & Documentation
- [ ] **Interactive Visuals & Demos:**
  - Record and embed an animated demo GIF / video in README.md showing the high-speed GPU HTML5 canvas terminal and 0ms stock switching in Google Colab.
  - Add architectural SVG / Mermaid diagrams illustrating the full data pipeline (Upstox API v2 ➔ TA Engine ➔ LightGBM ➔ Risk Manager ➔ Canvas Terminal).
- [ ] **Repository Badges:**
  - Add live status badges (build passing, Python 3.10+, License MIT, Open In Colab, Upstox API v2).

### 2. LinkedIn & Developer Community Reach
- [ ] **Social Article & Feed Release:**
  - Publish the prepared long-form technical article and viral feed post highlighting the architecture, single-command setup, and non-blocking asyncio event loop solution.
  - Share quantitative insights, backtest metrics, and learnings on handling concurrency in notebook environments.
- [ ] **GitHub Releases & Packaging:**
  - Tag semantic versions (1.0.0 stable release).
  - Prepare a 1-click Colab launcher badge linked directly to the main branch notebook.

---

*Last Updated: September 2026*
