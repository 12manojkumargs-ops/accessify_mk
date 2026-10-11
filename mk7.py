"""
Accessify: automated web accessibility auditor
------------------------------------------------
One-file hackathon application.

Dataset:
    AccessGuru Original_full_data.csv

Run:
    pip install streamlit pandas numpy scikit-learn requests beautifulsoup4 lxml
    streamlit run app.py

Optional:
    Put Original_full_data.csv in the same folder as this file.

What it does:
    1. Accepts a website URL.
    2. Downloads its HTML.
    3. Runs a practical accessibility scan.
    4. Uses the AccessGuru dataset to learn violation-impact scores
       and retrieve dataset-grounded recommendations.
    5. Produces an interpretable 0-100 accessibility screening score.
    6. Shows critical problems, WCAG-related guidance and code fixes.
    7. Lets the user download a report.

Important:
    This is an automated screening tool, NOT a certification of WCAG
    compliance. Real accessibility testing should include manual testing
    and testing with assistive technologies/screen-reader users.
"""

import re
import io
import html
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Accessify | Web Accessibility",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .hero { padding: 2rem 2.2rem; border-radius: 26px; background: linear-gradient(125deg,#001B5E 0%,#003B8E 42%,#0077C8 72%,#00A6FB 100%); color:#fff; box-shadow:0 18px 45px rgba(15,23,42,.25); margin-bottom:1.4rem; }
    .brand { font-size:3.8rem; font-weight:950; letter-spacing:.01em; line-height:1; }
    .project-name { font-size:1.45rem; font-weight:800; margin-top:.5rem; }
    .hero-desc { font-size:1.05rem; max-width:1000px; line-height:1.6; margin-top:1rem; color:#f8fafc; }
    .score-wrap { border-radius:28px; padding:1.5rem; color:#fff; text-align:center; background:linear-gradient(145deg,#001B5E,#003B8E); border:3px solid #E10600; box-shadow:0 15px 35px rgba(0,0,0,.3); }
    .score-number { font-size:5.2rem; font-weight:950; line-height:1; letter-spacing:-.05em; }
    .score-label { font-size:1.15rem; font-weight:900; text-transform:uppercase; letter-spacing:.08em; }
    .score-bar { height:16px; border-radius:99px; background:#374151; overflow:hidden; margin-top:1rem; }
    .score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#ef4444,#facc15,#22c55e); }
    .issue-card { padding:1rem 1.2rem; border-radius:14px; border:1px solid rgba(128,128,128,.25); margin-bottom:.8rem; }
    .small-muted { opacity:.68; font-size:.9rem; }
    .error-badge { display:inline-flex; align-items:center; gap:.5rem; background:linear-gradient(120deg,#E10600,#B00000); color:#fff; padding:.55rem .95rem; border-radius:999px; font-weight:850; box-shadow:0 5px 16px rgba(225,6,0,.25); margin:.2rem 0 1rem 0; }
    .risk-panel { border:1px solid rgba(225,6,0,.45); border-left:5px solid #E10600; border-radius:14px; padding:1rem 1.1rem; margin:.5rem 0 1rem 0; }
    .feature-panel { border-radius:14px; padding:1rem; background:linear-gradient(120deg,rgba(0,27,94,.10),rgba(0,166,251,.12)); border:1px solid rgba(0,119,200,.35); }
    .feature-title { scroll-margin-top:130px !important; display:inline-block; padding:.48rem .85rem; border-radius:10px; background:linear-gradient(110deg,#002B72,#0077C8); color:#fff; font-size:1.15rem; font-weight:850; border-left:4px solid #E10600; margin:.4rem 0 .55rem 0; }
    .feature-desc { font-size:.96rem; opacity:.82; margin:.1rem 0 .85rem 0; }
    .deduction-card { padding:.8rem .95rem; border-radius:12px; border:1px solid rgba(128,128,128,.28); background:rgba(128,128,128,.045); min-height:100px; margin-bottom:.75rem; }
    .side-nav a { display:block; text-decoration:none; padding:.48rem .62rem; border-radius:8px; margin:.3rem 0; color:inherit; font-size:.88rem; line-height:1.25; font-weight:750; border:1px solid rgba(0,119,200,.35); background:linear-gradient(110deg,rgba(0,27,94,.09),rgba(0,166,251,.12)); }
    .side-nav a:hover { border-color:#E10600; background:rgba(0,119,200,.15); }
    html { scroll-padding-top:130px !important; }
    [id="detected-problems"], [id="developer-recommendations"], [id="junk-alt-text"], [id="risk-predictor"], [id="explainable-ai"], .section-anchor { scroll-margin-top:130px !important; }
    .section-anchor h2 { margin-top:0; padding-top:.05rem; }

    /* ACCESSIFY MODERN UI OVERHAUL — presentation only */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --af-blue:#35b8ff; --af-blue2:#1769ff; --af-red:#ff365e; --af-ink:#071225; }
    html { scroll-behavior:smooth; }
    .stApp { font-family:'DM Sans',sans-serif; }
    [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 8% 2%,rgba(23,105,255,.10),transparent 30%),radial-gradient(ellipse at 92% 12%,rgba(255,54,94,.055),transparent 27%); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stMainBlockContainer"] { padding-top:1.8rem; padding-bottom:3rem; max-width:1500px; }
    h1,h2,h3 { font-family:'Space Grotesk','DM Sans',sans-serif !important; letter-spacing:-.035em !important; }
    h2 { font-weight:800 !important; }
    [data-testid="stSidebar"] { border-right:1px solid rgba(53,184,255,.18); }
    [data-testid="stSidebar"] > div:first-child { background:linear-gradient(180deg,rgba(15,36,72,.055),rgba(23,105,255,.025)); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height:1.55; }
    .hero { position:relative; isolation:isolate; overflow:hidden; min-height:280px; padding:2.6rem 2.8rem; border-radius:30px; background:linear-gradient(120deg,#061126 0%,#08285b 42%,#075bb5 72%,#0b91e8 100%); color:#fff; box-shadow:0 24px 70px rgba(0,54,130,.27),inset 0 1px 0 rgba(255,255,255,.22); margin-bottom:1.25rem; border:1px solid rgba(100,200,255,.32); }
    .hero:before { content:''; position:absolute; z-index:-1; width:360px;height:360px;right:-65px;top:-160px;border-radius:50%;background:radial-gradient(circle,rgba(53,184,255,.48),rgba(53,184,255,0) 68%);animation:af-pulse 7s ease-in-out infinite; }
    .hero:after { content:''; position:absolute;z-index:-1;width:260px;height:260px;right:16%;bottom:-205px;border-radius:50%;border:1px solid rgba(255,255,255,.2);box-shadow:0 0 0 24px rgba(255,255,255,.025),0 0 0 48px rgba(255,255,255,.02); }
    .hero-topline { display:inline-flex;align-items:center;gap:.5rem;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.09);backdrop-filter:blur(12px);border-radius:999px;padding:.42rem .8rem;font-size:.78rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:#e4f5ff;margin-bottom:1.3rem; }
    .hero-dot { display:inline-block;width:8px;height:8px;border-radius:50%;background:#48f0bd;box-shadow:0 0 14px #48f0bd; }
    .brand { font-family:'Space Grotesk',sans-serif;font-size:clamp(3.1rem,7vw,5.4rem);font-weight:700;letter-spacing:-.075em;line-height:.95;text-shadow:0 5px 30px rgba(0,0,0,.18); }
    .brand-mark { color:#71d4ff; }
    .project-name { font-size:clamp(1.15rem,2.6vw,1.65rem);font-weight:700;letter-spacing:-.025em;margin-top:1rem;color:#eaf7ff; }
    .hero-desc { font-size:1rem;max-width:760px;line-height:1.7;margin-top:.75rem;color:#dceeff; }
    .hero-pills { display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.3rem; }
    .hero-pill { display:inline-flex;align-items:center;gap:.4rem;padding:.45rem .72rem;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.075);border-radius:10px;font-size:.78rem;font-weight:700;color:#f3f9ff;backdrop-filter:blur(8px); }
    .hero-float { position:absolute;right:2.3rem;bottom:2rem;width:185px;padding:1rem 1.05rem;border-radius:18px;background:rgba(4,19,43,.58);border:1px solid rgba(160,221,255,.38);box-shadow:0 15px 38px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.12);backdrop-filter:blur(18px);animation:af-float 5s ease-in-out infinite;transform:rotate(2deg); }
    .hero-float-label { font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#b5dcff;font-weight:800; }
    .hero-float-number { font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:white;margin:.28rem 0; }
    .hero-float-sub { color:#b5cbe5;font-size:.75rem; }
    @keyframes af-float { 0%,100%{transform:translateY(0) rotate(2deg)}50%{transform:translateY(-9px) rotate(-1deg)} }
    @keyframes af-pulse { 0%,100%{transform:scale(1);opacity:.75}50%{transform:scale(1.14);opacity:1} }
    @media(max-width:850px) { .hero{padding:1.7rem 1.45rem;min-height:0}.hero-float{position:relative;right:auto;bottom:auto;margin-top:1.35rem;width:max-content;max-width:100%;animation:none}.brand{font-size:3.1rem}.hero-desc{font-size:.94rem} }
    /* Modern glass-like panels and interactive hover feedback */
    [data-testid="stMetric"] { padding:1rem 1.05rem;border-radius:17px;border:1px solid rgba(110,150,200,.22);background:linear-gradient(135deg,rgba(255,255,255,.72),rgba(231,242,255,.38));box-shadow:0 8px 24px rgba(16,54,105,.055);transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease; }
    [data-testid="stMetric"]:hover { transform:translateY(-4px);border-color:rgba(23,105,255,.45);box-shadow:0 14px 30px rgba(23,105,255,.12); }
    [data-testid="stMetricLabel"] { font-weight:700; }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif;font-weight:700; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px !important; }
    div[data-testid="stExpander"] { border:1px solid rgba(95,145,210,.25);border-radius:16px;background:rgba(120,160,220,.035); }
    div[data-testid="stExpander"] summary { font-weight:750; }
    .stButton > button, .stDownloadButton > button { border-radius:13px !important;min-height:2.8rem;font-weight:800 !important;letter-spacing:.005em;transition:transform .18s ease,box-shadow .18s ease,filter .18s ease !important; }
    .stButton > button[kind="primary"] { color:white !important;border:1px solid rgba(105,200,255,.5) !important;background:linear-gradient(110deg,#0759df,#087fc5 65%,#13a8df) !important;box-shadow:0 9px 23px rgba(0,103,230,.24) !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px);filter:brightness(1.06);box-shadow:0 12px 26px rgba(23,105,255,.18) !important; }
    .stTextInput input, .stTextArea textarea { border-radius:13px !important;border:1px solid rgba(100,150,210,.42) !important;min-height:3rem;transition:border-color .2s ease,box-shadow .2s ease; }
    .stTextInput input:focus, .stTextArea textarea:focus { border-color:#1688ff !important;box-shadow:0 0 0 3px rgba(22,136,255,.15) !important; }
    [data-testid="stAlert"] { border-radius:14px; }
    .issue-card, .deduction-card, .risk-panel, .feature-panel { transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
    .issue-card:hover, .deduction-card:hover { transform:translateY(-2px);border-color:rgba(23,105,255,.45);box-shadow:0 10px 24px rgba(16,54,105,.08); }
    .feature-title { box-shadow:0 7px 20px rgba(0,119,200,.15); }
    .side-nav a { padding:.65rem .75rem;border-radius:11px;transition:transform .16s ease,background .16s ease,border-color .16s ease; }
    .side-nav a:hover { transform:translateX(3px);border-color:#ff365e;background:linear-gradient(110deg,rgba(23,105,255,.13),rgba(255,54,94,.08)); }
    /* Keep text readable when Streamlit is in dark mode */
    @media(prefers-color-scheme:dark) {
      [data-testid="stMetric"] { background:linear-gradient(135deg,rgba(20,34,57,.95),rgba(13,26,46,.88));border-color:rgba(120,170,230,.25); }
      div[data-testid="stExpander"] { background:rgba(16,35,62,.35); }
    }

    /* ACCESSIFY: Red Bull Racing-inspired midnight / cyan / teal / ultramarine / racing-red / gold palette */
    :root { --af-navy:#06101f; --af-blue:#1769ff; --af-ultra:#4338ca; --af-cyan:#32d7ff; --af-teal:#25e0c1; --af-red:#e10600; }
    [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 5% 3%,rgba(23,105,255,.13),transparent 34%),radial-gradient(ellipse at 96% 7%,rgba(37,224,193,.08),transparent 27%),radial-gradient(ellipse at 78% 40%,rgba(225,6,0,.035),transparent 25%); }
    .hero { background:linear-gradient(112deg,#050b16 0%,#071b37 32%,#123c86 60%,#087eaa 83%,#0b9b91 100%); border:1px solid rgba(76,213,255,.38); box-shadow:0 22px 60px rgba(0,29,78,.24),0 0 28px rgba(23,105,255,.08),inset 0 1px 0 rgba(255,255,255,.15); animation:hero-arrive .75s cubic-bezier(.2,.8,.2,1) both; }
    .hero:before { background:radial-gradient(circle,rgba(50,215,255,.35),rgba(37,224,193,.14) 38%,transparent 70%); }
    .hero:after { border-color:rgba(50,215,255,.24); box-shadow:0 0 0 24px rgba(50,215,255,.025),0 0 0 48px rgba(37,224,193,.018); }
    .brand-mark { color:#48e3ff; text-shadow:0 0 24px rgba(50,215,255,.24); }
    .hero-topline { border-color:rgba(50,215,255,.28); background:linear-gradient(100deg,rgba(23,105,255,.15),rgba(37,224,193,.09)); }
    .hero-pill { border-color:rgba(99,218,255,.23); background:linear-gradient(120deg,rgba(23,105,255,.13),rgba(37,224,193,.06)); }
    .hero-float { border-color:rgba(50,215,255,.42); background:linear-gradient(145deg,rgba(5,17,37,.84),rgba(9,49,83,.68)); box-shadow:0 15px 38px rgba(0,0,0,.22),0 0 24px rgba(50,215,255,.08),inset 0 1px 0 rgba(255,255,255,.12); }
    .hero-float-label { color:#68e8ef; }
    .scan-intro { margin:1.25rem 0 .75rem; animation:hero-arrive .85s .08s both; }
    .scan-kicker { display:flex;align-items:center;gap:.55rem;color:#0c9db7;font-size:.74rem;font-weight:900;letter-spacing:.16em;margin-bottom:.55rem; }
    .scan-kicker-dot { width:8px;height:8px;border-radius:50%;background:#25e0c1;box-shadow:0 0 0 4px rgba(37,224,193,.13),0 0 14px rgba(37,224,193,.7); }
    .scan-title { font-family:'Space Grotesk','DM Sans',sans-serif;font-size:clamp(1.55rem,3.1vw,2.25rem);line-height:1.12;font-weight:800;letter-spacing:-.055em; }
    .scan-title span { background:linear-gradient(95deg,#1769ff 0%,#06a6d5 45%,#14b8a6 80%,#e10600 130%);-webkit-background-clip:text;background-clip:text;color:transparent; }
    .scan-copy { max-width:780px;font-size:.98rem;line-height:1.65;opacity:.76;margin-top:.55rem; }
    [class*="st-key-scan_input_panel"] { position:relative; overflow:hidden; margin:.7rem 0 .2rem; padding:1.05rem 1.1rem .85rem; border:1px solid rgba(50,165,225,.34); border-radius:20px; background:linear-gradient(120deg,rgba(23,105,255,.055),rgba(50,215,255,.065) 48%,rgba(37,224,193,.045)); box-shadow:0 12px 34px rgba(13,67,135,.07),inset 0 1px 0 rgba(255,255,255,.48); transition:border-color .25s ease,box-shadow .25s ease,transform .25s ease; animation:panel-arrive .7s .14s both; }
    [class*="st-key-scan_input_panel"]:hover { border-color:rgba(50,215,255,.65);box-shadow:0 16px 38px rgba(23,105,255,.10),0 0 24px rgba(50,215,255,.055);transform:translateY(-1px); }
    [class*="st-key-scan_input_panel"] [data-testid="stTextInput"] input { min-height:3.45rem!important;border-radius:13px!important;border:1px solid rgba(54,151,211,.45)!important;background:rgba(255,255,255,.88)!important;color:#071225!important;font-size:1.02rem!important;box-shadow:inset 0 1px 3px rgba(5,30,65,.04); }
    [class*="st-key-scan_input_panel"] [data-testid="stTextInput"] input:focus { border-color:#18bde7!important;box-shadow:0 0 0 3px rgba(50,215,255,.15),0 0 20px rgba(50,215,255,.12)!important; }
    [class*="st-key-scan_input_panel"] [data-testid="stFormSubmitButton"] button { min-height:3.15rem!important;border-radius:13px!important;color:#fff!important;border:1px solid rgba(84,220,255,.55)!important;background:linear-gradient(105deg,#0b3eaa 0%,#1769ff 38%,#079fca 72%,#0eaa98 100%)!important;box-shadow:0 8px 22px rgba(23,105,255,.2),inset 0 1px 0 rgba(255,255,255,.18)!important;transition:transform .2s ease,filter .2s ease,box-shadow .2s ease!important; }
    [class*="st-key-scan_input_panel"] [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px);filter:brightness(1.08);box-shadow:0 12px 28px rgba(23,105,255,.25),0 0 18px rgba(50,215,255,.12)!important; }
    .enter-hint { height:3.1rem;display:flex;flex-direction:column;justify-content:center;text-align:center;font-size:.7rem;font-weight:900;letter-spacing:.09em;opacity:.62;line-height:1.6; }
    .enter-hint span { font-weight:500;letter-spacing:0;font-size:.76rem; }
    .scan-footnote { margin:.25rem 0 1rem;font-size:.78rem;opacity:.63;letter-spacing:.01em; }
    .scan-footnote span { color:#0ca9c6;font-weight:900; }
    @keyframes hero-arrive { from { opacity:0;transform:translateY(12px) } to { opacity:1;transform:translateY(0) } }
    @keyframes panel-arrive { from { opacity:0;transform:translateY(9px) } to { opacity:1;transform:translateY(0) } }
    @media(prefers-color-scheme:dark) {
      [class*="st-key-scan_input_panel"] { background:linear-gradient(120deg,rgba(23,105,255,.13),rgba(50,215,255,.07) 48%,rgba(37,224,193,.06));box-shadow:0 12px 34px rgba(0,0,0,.14),inset 0 1px 0 rgba(255,255,255,.04); }
      .scan-kicker { color:#61dff0; }
      [class*="st-key-scan_input_panel"] [data-testid="stTextInput"] input { background:rgba(6,18,36,.9)!important;color:#f2f8ff!important;border-color:rgba(50,215,255,.35)!important; }
    }
    @media(max-width:650px) { [class*="st-key-scan_input_panel"] { padding:.8rem .75rem .6rem; } .scan-copy { font-size:.91rem; } }
    @media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important; } }


    /* NEON CYBER FINISH — restrained motion, racing red + cyan + ultramarine + gold */
    :root { --af-red:#e10600; --af-gold:#f6bd16; --af-cyan:#32d7ff; --af-teal:#25e0c1; --af-ultra:#4338ca; }
    .hero { background:linear-gradient(112deg,#050914 0%,#091a38 27%,#173d91 52%,#087eaa 75%,#087c72 100%); }
    .hero:before { background:radial-gradient(circle,rgba(50,215,255,.34),rgba(67,56,202,.20) 42%,rgba(225,6,0,.10) 60%,transparent 72%); animation:af-pulse 6.2s ease-in-out infinite, af-hue 12s linear infinite; }
    .hero:after { border-color:rgba(246,189,22,.24); box-shadow:0 0 0 24px rgba(50,215,255,.035),0 0 0 48px rgba(246,189,22,.025),0 0 0 72px rgba(225,6,0,.018); animation:orbit-breathe 8s ease-in-out infinite; }
    .brand-mark { background:linear-gradient(95deg,#32d7ff,#25e0c1 42%,#f6bd16 76%,#ff594d);-webkit-background-clip:text;background-clip:text;color:transparent;text-shadow:none; }
    .hero-topline { box-shadow:0 0 22px rgba(50,215,255,.08); }
    .hero-pill { transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease,background .22s ease; }
    .hero-pill:hover { transform:translateY(-3px);border-color:rgba(246,189,22,.65);box-shadow:0 8px 24px rgba(50,215,255,.12),0 0 12px rgba(246,189,22,.08); }
    .hero-float { animation:af-float 5s ease-in-out infinite, float-glow 4.5s ease-in-out infinite; }
    .scan-title span { background:linear-gradient(95deg,#1769ff 0%,#32d7ff 33%,#25e0c1 58%,#f6bd16 80%,#e10600 110%);background-size:220% auto;animation:gradient-drift 7s linear infinite;-webkit-background-clip:text;background-clip:text;color:transparent; }
    [class*="st-key-scan_input_panel"] { background:linear-gradient(125deg,rgba(23,105,255,.075),rgba(50,215,255,.075) 38%,rgba(37,224,193,.055) 66%,rgba(225,6,0,.035)); }
    [class*="st-key-scan_input_panel"]:before { content:"";pointer-events:none;position:absolute;inset:0;border-radius:inherit;background:linear-gradient(115deg,transparent 15%,rgba(50,215,255,.07) 42%,transparent 64%);background-size:220% 100%;animation:panel-sheen 8s ease-in-out infinite; }
    [class*="st-key-scan_input_panel"] [data-testid="stFormSubmitButton"] button { background:linear-gradient(105deg,#092c84 0%,#1769ff 30%,#079fca 58%,#0eaa98 78%,#d43b28 115%)!important;background-size:180% 100%!important; }
    [class*="st-key-scan_input_panel"] [data-testid="stFormSubmitButton"] button:hover { background-position:100% 0!important;box-shadow:0 12px 28px rgba(23,105,255,.25),0 0 22px rgba(50,215,255,.18),0 0 12px rgba(225,6,0,.10)!important; }
    [data-testid="stMetric"] { position:relative;overflow:hidden;transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease!important; }
    [data-testid="stMetric"]:after { content:"";position:absolute;left:-110%;top:0;width:70%;height:2px;background:linear-gradient(90deg,transparent,#32d7ff,#f6bd16,transparent);transition:left .55s ease; }
    [data-testid="stMetric"]:hover:after { left:130%; }
    [data-testid="stMetric"]:hover { transform:translateY(-5px) scale(1.008)!important;border-color:rgba(50,215,255,.48)!important;box-shadow:0 14px 32px rgba(23,105,255,.12),0 0 18px rgba(50,215,255,.055)!important; }
    .issue-card:hover,.deduction-card:hover { border-color:rgba(50,215,255,.55);box-shadow:0 10px 25px rgba(23,105,255,.10),0 0 15px rgba(50,215,255,.055); }
    .feature-title { background:linear-gradient(105deg,#071b44,#123c86 55%,#087eaa);border-left-color:#e10600;box-shadow:0 7px 22px rgba(23,105,255,.16),0 0 15px rgba(50,215,255,.045); }
    .error-badge { box-shadow:0 5px 18px rgba(225,6,0,.22),0 0 14px rgba(225,6,0,.09); }
    .risk-panel { background:linear-gradient(115deg,rgba(23,105,255,.055),rgba(50,215,255,.035),rgba(225,6,0,.025)); }
    @keyframes orbit-breathe { 0%,100%{transform:scale(1) rotate(0deg);opacity:.7}50%{transform:scale(1.04) rotate(8deg);opacity:1} }
    @keyframes float-glow { 0%,100%{box-shadow:0 15px 38px rgba(0,0,0,.22),0 0 15px rgba(50,215,255,.06),inset 0 1px 0 rgba(255,255,255,.12)}50%{box-shadow:0 18px 42px rgba(0,0,0,.25),0 0 25px rgba(50,215,255,.15),0 0 12px rgba(246,189,22,.05),inset 0 1px 0 rgba(255,255,255,.15)} }
    @keyframes gradient-drift { 0%{background-position:0% 50%}100%{background-position:220% 50%} }
    @keyframes panel-sheen { 0%,65%,100%{background-position:180% 0}35%{background-position:-20% 0} }
    @keyframes af-hue { 0%,100%{filter:hue-rotate(0deg)}50%{filter:hue-rotate(18deg)} }

    /* Streamlit's own theme selector is more reliable than OS theme preference. */
    [data-testid="stAppViewContainer"][data-theme="light"], [data-theme="light"] [data-testid="stAppViewContainer"] {
      color:#111827;
      background:radial-gradient(ellipse at 5% 3%,rgba(23,105,255,.075),transparent 34%),radial-gradient(ellipse at 96% 7%,rgba(37,224,193,.055),transparent 27%),#f5f8ff;
    }
    [data-testid="stAppViewContainer"][data-theme="light"] [data-testid="stMetric"], [data-theme="light"] [data-testid="stMetric"] { background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(235,244,255,.92))!important;color:#111827!important;border-color:rgba(23,105,255,.20)!important; }
    [data-theme="light"] .feature-desc,[data-theme="light"] .scan-copy,[data-theme="light"] .scan-footnote { color:#344256!important;opacity:1!important; }
    [data-theme="light"] .deduction-card,[data-theme="light"] .issue-card { background:rgba(255,255,255,.84)!important;color:#111827!important;border-color:rgba(70,95,130,.22)!important; }
    [data-theme="light"] div[data-testid="stExpander"] { background:rgba(255,255,255,.88)!important;color:#111827!important;border-color:rgba(70,95,130,.20)!important; }
    [data-theme="light"] [data-testid="stSidebar"] { background:#eef4ff!important;color:#111827!important; }
    [data-theme="light"] [class*="st-key-scan_input_panel"] { background:linear-gradient(120deg,rgba(23,105,255,.075),rgba(50,215,255,.09) 48%,rgba(37,224,193,.06));border-color:rgba(23,105,255,.24); }
    [data-theme="light"] [class*="st-key-scan_input_panel"] [data-testid="stTextInput"] input { background:#fff!important;color:#071225!important;border-color:rgba(23,105,255,.34)!important; }
    [data-theme="dark"] [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 5% 3%,rgba(23,105,255,.14),transparent 34%),radial-gradient(ellipse at 96% 7%,rgba(37,224,193,.08),transparent 27%),#07101e; }
    [data-theme="dark"] [data-testid="stMetric"] { background:linear-gradient(135deg,rgba(16,31,54,.97),rgba(9,22,41,.94))!important;color:#f2f7ff!important;border-color:rgba(50,215,255,.20)!important; }
    [data-theme="dark"] .deduction-card,[data-theme="dark"] .issue-card { background:rgba(12,28,49,.76)!important;color:#f2f7ff!important;border-color:rgba(105,155,210,.25)!important; }
    [data-theme="dark"] div[data-testid="stExpander"] { background:rgba(10,25,45,.7)!important;color:#f2f7ff!important;border-color:rgba(50,215,255,.18)!important; }
    [data-theme="dark"] [data-testid="stSidebar"] { background:#081426!important;color:#f2f7ff!important; }
    [data-theme="dark"] .feature-desc,[data-theme="dark"] .scan-copy,[data-theme="dark"] .scan-footnote { color:#c3d2e8!important;opacity:1!important; }
    @media (hover:none) { .hero-pill:hover,[data-testid="stMetric"]:hover,.issue-card:hover,.deduction-card:hover { transform:none!important; } }

    /* DATA TABLE LEGIBILITY — especially dataset/results tables in light mode */
    [data-testid="stDataFrame"] {
      --gdg-bg-cell:#ffffff;
      --gdg-bg-cell-medium:#f3f7ff;
      --gdg-bg-header:#e6efff;
      --gdg-bg-header-hovered:#d7e7ff;
      --gdg-text-dark:#142238;
      --gdg-text-medium:#2f4059;
      --gdg-text-light:#52647c;
      --gdg-border-color:#cbd9ed;
      border:1px solid rgba(23,105,255,.22)!important;
      border-radius:14px!important;
      overflow:hidden!important;
      box-shadow:0 8px 22px rgba(22,55,100,.06);
      transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;
    }
    [data-testid="stDataFrame"]:hover {
      transform:translateY(-2px);
      border-color:rgba(50,215,255,.55)!important;
      box-shadow:0 12px 28px rgba(23,105,255,.10),0 0 16px rgba(50,215,255,.06);
    }
    [data-theme="light"] [data-testid="stDataFrame"],
    [data-testid="stAppViewContainer"][data-theme="light"] [data-testid="stDataFrame"] {
      --gdg-bg-cell:#ffffff;
      --gdg-bg-cell-medium:#f4f8ff;
      --gdg-bg-header:#e6efff;
      --gdg-bg-header-hovered:#d7e7ff;
      --gdg-text-dark:#142238;
      --gdg-text-medium:#2f4059;
      --gdg-text-light:#52647c;
      --gdg-border-color:#cbd9ed;
      background:#ffffff!important;
      color:#142238!important;
      border-color:rgba(23,105,255,.22)!important;
    }
    [data-theme="dark"] [data-testid="stDataFrame"] {
      --gdg-bg-cell:#0d1b2e;
      --gdg-bg-cell-medium:#12243b;
      --gdg-bg-header:#172d49;
      --gdg-bg-header-hovered:#203d60;
      --gdg-text-dark:#f2f7ff;
      --gdg-text-medium:#d4e2f5;
      --gdg-text-light:#a9bfdc;
      --gdg-border-color:#29425f;
      background:#0d1b2e!important;
      border-color:rgba(50,215,255,.20)!important;
    }

    /* Subtle cursor-responsive feel: cards lift and tilt slightly on hover */
    [data-testid="stMetric"],.issue-card,.deduction-card,.feature-panel,
    .risk-panel,.hero-pill,[class*="st-key-scan_input_panel"] {
      transform-style:preserve-3d;
      will-change:transform;
    }
    [data-testid="stMetric"]:hover {
      transform:perspective(850px) translateY(-5px) rotateX(.6deg) rotateY(-.5deg) scale(1.008)!important;
    }
    .issue-card:hover,.deduction-card:hover,.feature-panel:hover,.risk-panel:hover {
      transform:perspective(900px) translateY(-3px) rotateX(.35deg) rotateY(-.25deg);
      box-shadow:0 12px 28px rgba(23,105,255,.10),0 0 16px rgba(50,215,255,.06);
    }
    .stTextInput input:hover,.stTextArea textarea:hover,.stSelectbox [data-baseweb="select"]:hover {
      border-color:rgba(50,215,255,.7)!important;
      box-shadow:0 0 0 2px rgba(50,215,255,.07),0 0 15px rgba(50,215,255,.07);
    }
    .stButton > button:hover,.stDownloadButton > button:hover { transform:translateY(-2px) scale(1.012); }
    @media(hover:none) {
      .hero-pill:hover,[data-testid="stMetric"]:hover,.issue-card:hover,.deduction-card:hover,
      .feature-panel:hover,.risk-panel:hover,[data-testid="stDataFrame"]:hover { transform:none!important; }
    }
    @media(prefers-reduced-motion:reduce) {
      [data-testid="stDataFrame"],[data-testid="stMetric"],.issue-card,.deduction-card,
      .feature-panel,.risk-panel,.hero-pill,.stButton > button { transition:none!important; }
    }

    /* V4: cursor-reactive depth, soft neon spotlight, and tactile controls */
    .af-cursor-active[data-testid="stMetric"],
    .af-cursor-active.issue-card,
    .af-cursor-active.deduction-card,
    .af-cursor-active.feature-panel,
    .af-cursor-active.risk-panel,
    .af-cursor-active[data-testid="stDataFrame"],
    .af-cursor-active.hero-pill {
      transform:perspective(950px) translate3d(var(--af-tx,0px),var(--af-ty,-3px),0)
        rotateX(var(--af-rx,0deg)) rotateY(var(--af-ry,0deg)) scale(1.008)!important;
      border-color:rgba(50,215,255,.58)!important;
      box-shadow:0 16px 34px rgba(23,105,255,.14),0 0 20px rgba(50,215,255,.10),inset 0 1px 0 rgba(255,255,255,.18)!important;
    }
    .af-cursor-active.stButton button,
    .af-cursor-active.stDownloadButton button,
    button.af-cursor-active {
      transform:translate3d(var(--af-tx,0px),var(--af-ty,-2px),0) scale(1.015)!important;
      filter:brightness(1.09)!important;
      box-shadow:0 12px 28px rgba(23,105,255,.22),0 0 20px rgba(50,215,255,.13)!important;
    }
    .af-cursor-active [data-testid="stTextInput"] input,
    .af-cursor-active input { box-shadow:0 0 0 2px rgba(50,215,255,.10),0 0 22px rgba(50,215,255,.12)!important; }
    .af-cursor-orb {
      position:fixed;width:96px;height:96px;border-radius:50%;pointer-events:none;z-index:999999;
      background:radial-gradient(circle,rgba(50,215,255,.12) 0%,rgba(23,105,255,.055) 34%,rgba(67,56,202,.025) 54%,transparent 72%);
      transform:translate(-50%,-50%);opacity:0;transition:opacity .22s ease;mix-blend-mode:screen;
    }
    .af-cursor-orb.af-visible { opacity:1; }
    /* Keep the hero's decorative glow stationary; cursor effects remain on other elements. */
    .hero:after { pointer-events:none; }
    .hero, [class*="st-key-scan_input_panel"] { transition:box-shadow .28s ease,border-color .28s ease; }
    .af-scan-sheen { position:relative; }
    [data-theme="light"] .af-cursor-orb { mix-blend-mode:multiply;opacity:.38; }
    @media(hover:none), (prefers-reduced-motion:reduce) {
      .af-cursor-orb { display:none!important; }
      .af-cursor-active[data-testid="stMetric"],.af-cursor-active.issue-card,
      .af-cursor-active.deduction-card,.af-cursor-active.feature-panel,
      .af-cursor-active.risk-panel,.af-cursor-active[data-testid="stDataFrame"],
      .af-cursor-active.hero-pill,.af-cursor-active.stButton button,
      .af-cursor-active.stDownloadButton button,button.af-cursor-active { transform:none!important; }
      .hero:before { left:auto;top:auto;right:-65px;bottom:auto;transform:none; }
    }

    /* V5: animated racing-gradient titles and richer cursor-reactive glow */
    :root { --af-racing-gradient:linear-gradient(100deg,#315bff 0%,#5b35ff 19%,#00c8ff 38%,#f6bd16 59%,#ff3b45 76%,#315bff 100%); }
    .brand-mark,.scan-title span,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 {
      background-image:var(--af-racing-gradient)!important;
      background-size:260% 100%!important;
      background-position:0% 50%;
      animation:af-title-race 6.2s linear infinite;
      -webkit-background-clip:text!important;background-clip:text!important;
      -webkit-text-fill-color:transparent!important;color:transparent!important;
      text-shadow:0 0 17px rgba(49,91,255,.10);
      transition:background-position .22s ease,filter .22s ease,text-shadow .22s ease;
    }
    .brand-mark { filter:drop-shadow(0 0 8px rgba(0,200,255,.18)); }
    .scan-title span { filter:drop-shadow(0 0 9px rgba(0,200,255,.16)); }
    .feature-title {
      position:relative;isolation:isolate;display:inline-block;
      color:#f4f8ff!important;-webkit-text-fill-color:#f4f8ff!important;
      background:none!important;background-color:transparent!important;background-image:none!important;
      border-left:0!important;padding:.62rem 1rem .62rem 1.12rem!important;
      filter:drop-shadow(0 0 9px rgba(49,91,255,.16));
      text-shadow:0 0 10px rgba(50,215,255,.12);
    }
    .feature-title:before {
      content:"";position:absolute;inset:0;z-index:-1;border-radius:11px;
      background:linear-gradient(110deg,#071635 0%,#162c69 38%,#07566d 68%,#38215f 100%);
      background-size:220% 100%;animation:af-title-race 8s linear infinite;
      border:1px solid rgba(50,215,255,.58);
      box-shadow:0 0 0 1px rgba(67,56,202,.18),0 7px 22px rgba(23,105,255,.22),0 0 22px rgba(50,215,255,.13),0 0 12px rgba(246,189,22,.08);
      transition:box-shadow .22s ease,border-color .22s ease;
    }
    [data-theme="light"] .feature-title { color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;text-shadow:0 1px 2px rgba(0,0,0,.28),0 0 10px rgba(50,215,255,.16)!important; }
    [data-theme="dark"] .feature-title { color:#ffffff!important;-webkit-text-fill-color:#ffffff!important; }
    .feature-title:after {
      content:"";position:absolute;left:0;top:14%;height:72%;width:3px;border-radius:3px;
      background:linear-gradient(180deg,#32d7ff,#f6bd16,#e10600);box-shadow:0 0 12px rgba(50,215,255,.48);
    }
    .af-text-active.brand-mark,.af-text-active.scan-title span,
    .stMarkdown h1.af-text-active,.stMarkdown h2.af-text-active,.stMarkdown h3.af-text-active {
      animation-play-state:paused!important;
      background-position:var(--af-text-x,75%) 50%!important;
      filter:drop-shadow(0 0 10px rgba(0,200,255,.36)) drop-shadow(0 0 16px rgba(246,189,22,.20))!important;
      text-shadow:0 0 13px rgba(0,200,255,.34),0 0 24px rgba(246,189,22,.14),0 0 12px rgba(225,6,0,.12)!important;
    }
    .feature-title:hover:before,.af-text-active.feature-title:before {
      border-color:rgba(246,189,22,.82);
      box-shadow:0 0 0 1px rgba(67,56,202,.25),0 8px 26px rgba(23,105,255,.30),0 0 28px rgba(50,215,255,.24),0 0 18px rgba(246,189,22,.15),0 0 12px rgba(225,6,0,.10);
    }
    .af-text-active.feature-title { color:#fff!important;-webkit-text-fill-color:#fff!important; }
    .hero { box-shadow:0 24px 70px rgba(0,54,130,.30),0 0 30px rgba(23,105,255,.16),0 0 14px rgba(225,6,0,.07),inset 0 1px 0 rgba(255,255,255,.22); }
    .hero-float { box-shadow:0 18px 44px rgba(0,0,0,.26),0 0 26px rgba(50,215,255,.17),0 0 13px rgba(246,189,22,.08),inset 0 1px 0 rgba(255,255,255,.14); }
    .hero-pill:hover { box-shadow:0 9px 25px rgba(50,215,255,.20),0 0 16px rgba(246,189,22,.13),0 0 12px rgba(225,6,0,.08)!important; }
    [data-testid="stMetric"]:hover,.issue-card:hover,.deduction-card:hover,.feature-panel:hover,.risk-panel:hover,
    [data-testid="stDataFrame"]:hover {
      box-shadow:0 16px 36px rgba(23,105,255,.17),0 0 24px rgba(50,215,255,.13),0 0 12px rgba(246,189,22,.055)!important;
    }
    .stButton > button:hover,.stDownloadButton > button:hover {
      box-shadow:0 12px 30px rgba(23,105,255,.27),0 0 24px rgba(50,215,255,.18),0 0 13px rgba(225,6,0,.12)!important;
    }
    [data-theme="light"] .stMarkdown h1,[data-theme="light"] .stMarkdown h2,[data-theme="light"] .stMarkdown h3 {
      background-image:linear-gradient(100deg,#173eb9 0%,#4338ca 20%,#007baf 40%,#a56a00 60%,#c51f36 78%,#173eb9 100%)!important;
    }
    [data-theme="light"] .stMarkdown h1.af-text-active,[data-theme="light"] .stMarkdown h2.af-text-active,[data-theme="light"] .stMarkdown h3.af-text-active {
      filter:drop-shadow(0 0 8px rgba(23,105,255,.22)) drop-shadow(0 0 10px rgba(197,31,54,.12))!important;
    }
    @keyframes af-title-race { 0%{background-position:0% 50%}100%{background-position:260% 50%} }
    @media(prefers-reduced-motion:reduce) {
      .brand-mark,.scan-title span,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 { animation:none!important; }
      .af-text-active.brand-mark,.af-text-active.scan-title span,
      .stMarkdown h1.af-text-active,.stMarkdown h2.af-text-active,.stMarkdown h3.af-text-active { filter:none!important;text-shadow:none!important; }
      .feature-title:before { animation:none!important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CURSOR INTERACTION LAYER — UI ONLY
# ============================================================
# Runs in a tiny component and enhances the parent Streamlit page.
# If the browser blocks parent access, normal CSS hover states still work.
components.html(
    """
    <script>
    (() => {
      try {
        const doc = window.parent.document;
        const win = window.parent;
        if (doc.__accessifyCursorLayerV5) return;
        doc.__accessifyCursorLayerV5 = true;

        const style = doc.createElement('style');
        style.textContent = `
          @media (hover:hover) and (pointer:fine) {
            .af-cursor-orb { display:block; }
            .hero { transition:box-shadow .25s ease,border-color .25s ease; }
          }
        `;
        doc.head.appendChild(style);

        const orb = doc.createElement('div');
        orb.className = 'af-cursor-orb';
        orb.setAttribute('aria-hidden', 'true');
        // Compact neon cursor glow; the hero title spotlight stays stationary.
        doc.body.appendChild(orb);

        const cardSelector = '[data-testid="stMetric"], .issue-card, .deduction-card, .feature-panel, .risk-panel, [data-testid="stDataFrame"], .hero-pill';
        const buttonSelector = '.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button';
        const inputPanelSelector = '[class*="st-key-scan_input_panel"]';
        const interactiveSelector = `${cardSelector}, ${buttonSelector}, ${inputPanelSelector}`;
        const textSelector = '.brand-mark, .scan-title span, .feature-title, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3';
        let active = null;
        let activeText = null;
        let frame = 0;
        let lastX = 0, lastY = 0;

        const clearActive = () => {
          if (!active) return;
          active.classList.remove('af-cursor-active');
          ['--af-tx','--af-ty','--af-rx','--af-ry'].forEach(k => active.style.removeProperty(k));
          active = null;
        };
        const clearActiveText = () => {
          if (!activeText) return;
          activeText.classList.remove('af-text-active');
          activeText.style.removeProperty('--af-text-x');
          activeText = null;
        };

        doc.addEventListener('pointerover', (event) => {
          if (event.pointerType && event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;
          const next = event.target && event.target.closest ? event.target.closest(interactiveSelector) : null;
          if (next !== active) {
            clearActive();
            active = next;
            if (active) active.classList.add('af-cursor-active');
          }
          const nextText = event.target && event.target.closest ? event.target.closest(textSelector) : null;
          if (nextText !== activeText) {
            clearActiveText();
            activeText = nextText;
            if (activeText) activeText.classList.add('af-text-active');
          }
        }, true);

        doc.addEventListener('pointermove', (event) => {
          if (event.pointerType && event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;
          lastX = event.clientX; lastY = event.clientY;
          if (!frame) frame = win.requestAnimationFrame(() => {
            frame = 0;
            orb.style.left = `${lastX}px`;
            orb.style.top = `${lastY}px`;
            orb.classList.add('af-visible');
            if (activeText && activeText.isConnected) {
              const textRect = activeText.getBoundingClientRect();
              const textDX = Math.max(0, Math.min(1, (lastX - textRect.left) / Math.max(1, textRect.width)));
              activeText.style.setProperty('--af-text-x', `${Math.round(textDX * 100)}%`);
            }
            // Hero title spotlight intentionally stays stationary; other cursor effects remain active.
            if (active && active.isConnected) {
              const rect = active.getBoundingClientRect();
              const dx = Math.max(-1, Math.min(1, (lastX - rect.left) / Math.max(1, rect.width) * 2 - 1));
              const dy = Math.max(-1, Math.min(1, (lastY - rect.top) / Math.max(1, rect.height) * 2 - 1));
              const isButton = active.matches(buttonSelector);
              active.style.setProperty('--af-tx', `${dx * (isButton ? 3 : 2.5)}px`);
              active.style.setProperty('--af-ty', `${dy * (isButton ? 2 : 1.5) - (isButton ? 1.5 : 2)}px`);
              active.style.setProperty('--af-rx', `${-dy * (isButton ? 0 : 3)}deg`);
              active.style.setProperty('--af-ry', `${dx * (isButton ? 0 : 3)}deg`);
            }
          });
        }, {passive:true, capture:true});

        doc.addEventListener('pointerout', (event) => {
          if (event.target === doc.documentElement || event.target === doc.body) {
            orb.classList.remove('af-visible');
            clearActive();
            clearActiveText();
          }
          if (active && event.target === active && !active.contains(event.relatedTarget)) clearActive();
          if (activeText && event.target === activeText && !activeText.contains(event.relatedTarget)) clearActiveText();
        }, true);
        win.addEventListener('blur', () => { orb.classList.remove('af-visible'); clearActive(); clearActiveText(); });

        // Streamlit rerenders components during form submissions; the delegated listeners
        // keep working, and a tiny observer removes stale inline styles from detached nodes.
        const observer = new MutationObserver(() => {
          if (active && !active.isConnected) clearActive();
        });
        observer.observe(doc.body, {childList:true,subtree:true});
      } catch (error) {
        // Safe fallback: the app's CSS hover effects remain available.
      }
    })();
    </script>
    """,
    height=1,
    scrolling=False,
)

# ============================================================
# CONSTANTS
# ============================================================

DATASET_NAME = "Original_full_data.csv"

DEFAULT_DATASET_PATHS = [
    DATASET_NAME,
    "./data/Original_full_data.csv",
    "/mnt/data/Original_full_data.csv",
]

SEVERITY_WEIGHTS = {
    5: 5.0,  # critical
    4: 4.0,  # serious
    3: 3.0,  # moderate
    2: 2.0,  # minor
    1: 1.0,
}


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_accessguru_dataset(path):
    df = pd.read_csv(path, low_memory=False)

    # Normalize column names
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]

    required = [
        "violationnumberID",
        "initialImpactScore",
        "Description",
        "AffectedHTMLElement(s)",
        "category",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "The AccessGuru CSV is missing expected columns: "
            + ", ".join(missing)
        )

    df["initialImpactScore"] = pd.to_numeric(
        df["initialImpactScore"],
        errors="coerce",
    )

    df = df.dropna(subset=["initialImpactScore"]).copy()

    # Build text used by the ML model
    text_columns = [
        "violationnumberID",
        "Description",
        "AffectedHTMLElement(s)",
        "category",
        "LongDescription",
        "updated_Description",
    ]

    for col in text_columns:
        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].fillna("").astype(str)

    df["model_text"] = (
        df["violationnumberID"]
        + " "
        + df["Description"]
        + " "
        + df["AffectedHTMLElement(s)"]
        + " "
        + df["category"]
        + " "
        + df["LongDescription"]
    )

    return df


def find_dataset():
    for path in DEFAULT_DATASET_PATHS:
        try:
            with open(path, "rb"):
                return path
        except Exception:
            pass

    return None


# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_accessguru_model(df):
    """
    Learn the relationship between an accessibility violation's
    textual/HTML description and its AccessGuru impact score.

    This is deliberately interpretable and lightweight:
        TF-IDF -> Ridge regression

    The live scanner then uses semantic similarity against the
    AccessGuru corpus to retrieve the closest known violations and
    their fixes.
    """

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["model_text"])

    y = df["initialImpactScore"].astype(float).values

    model = Ridge(alpha=2.0)
    model.fit(X, y)

    return vectorizer, model


def risk_text_features(description, element, category, category_frequency=0):
    """Convert an accessibility finding into a compact tabular feature row."""
    description = str(description or "").lower()
    element = str(element or "").lower()
    combined = description + " " + element
    tags = re.findall(r"<\s*[a-z][a-z0-9-]*", element)
    keywords = [
        "alt", "label", "contrast", "keyboard", "color",
        "heading", "aria", "form", "video", "button", "image",
        "focus", "name", "link", "caption",
    ]
    row = [
        len(description),
        len(element),
        len(tags),
        len(re.findall(r"[a-z0-9]+", description)),
        float(category_frequency),
    ]
    row.extend(1 if word in combined else 0 for word in keywords)
    return row


RISK_FEATURE_NAMES = [
    "Description length", "HTML snippet length", "HTML tag count",
    "Description token count", "Category frequency", "Alt-related pattern",
    "Label-related pattern", "Contrast-related pattern", "Keyboard-related pattern",
    "Color-related pattern", "Heading-related pattern", "ARIA-related pattern",
    "Form-related pattern", "Video-related pattern", "Button-related pattern",
    "Image-related pattern", "Focus-related pattern", "Accessible-name pattern",
    "Link-related pattern", "Caption-related pattern",
]


@st.cache_resource(show_spinner=False)
def train_tabular_risk_model(df):
    """Train a small decision tree on AccessGuru-derived tabular text features."""
    categories = df["category"].fillna("").astype(str).str.lower()
    category_frequency = categories.value_counts(normalize=True).to_dict()
    rows = []
    labels = []
    for _, record in df.iterrows():
        description = record.get("Description", "")
        element = record.get("AffectedHTMLElement(s)", "")
        category = str(record.get("category", "")).lower()
        rows.append(risk_text_features(
            description, element, category, category_frequency.get(category, 0)
        ))
        impact = float(record["initialImpactScore"])
        if impact <= 2:
            labels.append("Low")
        elif impact < 4:
            labels.append("Moderate")
        elif impact < 5:
            labels.append("High")
        else:
            labels.append("Critical")
    model = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42
    )
    model.fit(np.asarray(rows, dtype=float), np.asarray(labels))
    return model, category_frequency


@st.cache_resource(show_spinner=False)
def train_alt_text_nlp_classifier():
    """Train a lightweight TF-IDF + Logistic Regression text classifier.

    Training labels are weakly supervised from curated examples. This provides
    an NLP-based screening signal, not a validated judgement of image meaning.
    """
    generic_examples = [
        "image", "img", "photo", "picture", "graphic", "image of",
        "picture of", "photo of", "click here", "untitled", "placeholder",
        "image123.jpg", "img_001.png", "DSC00421.JPG", "screenshot.png",
        "logo.png", "photo.jpg", "image dot png", "a picture", "an image",
        "banner image", "icon", "image file", "download", "IMG_2024.jpeg",
        "photo 1", "picture 2", "graphic file", "test image",
    ]
    descriptive_examples = [
        "A golden retriever running across a grassy field",
        "VIT Vellore campus entrance with students walking",
        "Search the website", "Open the navigation menu",
        "Red Bull racing car on the track during a race",
        "Bar chart comparing monthly energy consumption",
        "A student using a laptop in a classroom",
        "Company logo for Accessify accessibility checker",
        "Close dialog", "Submit registration form",
        "Map showing the route from the station to campus",
        "Doctor speaking with a patient in a clinic",
        "Blue accessibility icon showing a person in a wheelchair",
        "Graph showing website accessibility scores by category",
        "A person reading text on a computer screen",
        "Next page", "Download the annual report PDF",
        "Students collaborating on a science project",
        "Solar panel mounted on a rooftop under sunlight",
        "The homepage logo linking to the site homepage",
        "Magnifying glass icon for searching articles",
        "A line chart showing temperature increasing over time",
    ]
    texts = generic_examples + descriptive_examples
    labels = (["likely_unhelpful"] * len(generic_examples) +
              ["potentially_descriptive"] * len(descriptive_examples))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(texts, labels)
    return model


def detect_junk_alt_text(soup):
    """Use an NLP classifier plus transparent checks to flag suspicious alt text."""
    model = train_alt_text_nlp_classifier()
    findings = []
    generic_phrases = {
        "image", "img", "photo", "photograph", "picture", "graphic",
        "image of", "picture of", "photo of", "an image", "a picture",
        "click here", "untitled", "placeholder", "icon",
    }
    file_pattern = re.compile(
        r"^(?:img|image|photo|pic|dsc|screenshot)[_ -]?[a-z0-9_-]*\.(?:png|jpe?g|gif|webp|svg)$",
        re.I,
    )
    for img in soup.find_all("img"):
        if not img.has_attr("alt"):
            continue  # Missing alt is handled by the existing scanner.
        alt = re.sub(r"\s+", " ", str(img.get("alt", ""))).strip()
        if not alt:
            continue  # Empty alt may be correct for decorative images.
        normalized = re.sub(r"[^a-z0-9. ]+", " ", alt.lower()).strip()
        prediction = model.predict([alt])[0]
        probabilities = model.predict_proba([alt])[0]
        classes = list(model.classes_)
        unhelpful_probability = float(probabilities[classes.index("likely_unhelpful")])
        reasons = []
        if normalized in generic_phrases:
            reasons.append("generic label that does not describe the image")
        if file_pattern.match(alt):
            reasons.append("filename-like alt text")
        if re.search(r"\.(png|jpe?g|gif|webp|svg)$", alt, re.I):
            reasons.append("file extension exposed as the alternative text")
        if prediction == "likely_unhelpful" and unhelpful_probability >= 0.55:
            reasons.append(f"NLP classifier flagged this text as potentially unhelpful ({unhelpful_probability:.0%} model confidence)")
        if reasons:
            findings.append({
                "alt": alt,
                "src": str(img.get("src", ""))[:300],
                "reasons": list(dict.fromkeys(reasons)),
                "element": str(img)[:1200],
                "nlp_prediction": prediction,
                "nlp_confidence": unhelpful_probability,
            })
    return findings


@st.cache_resource(show_spinner=False)
def build_dataset_search_index(df):
    """
    Separate TF-IDF index used to find the closest AccessGuru
    examples for a detected problem.
    """

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["model_text"])

    return vectorizer, X


# ============================================================
# WEB FETCHING
# ============================================================

def normalize_url(url):
    url = url.strip()

    if not url:
        return None

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.netloc:
        return None

    return url


def fetch_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Accessify/1.0; "
            "+https://example.com/accessibility)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if "text/html" not in content_type and response.text.count("<") < 10:
        raise ValueError(
            "The URL did not return a normal HTML webpage."
        )

    return response.text, response.url


# ============================================================
# ACCESSIBILITY HELPERS
# ============================================================

def visible_text(element):
    return element.get_text(" ", strip=True)


def has_accessible_name(element, soup):
    """
    Practical accessible-name heuristic.

    It checks common sources:
      - visible text
      - aria-label
      - aria-labelledby
      - title
      - alt
      - associated label
    """

    if element.get("aria-label", "").strip():
        return True

    labelledby = element.get("aria-labelledby", "").strip()

    if labelledby:
        ids = labelledby.split()
        if any(soup.find(id=i) for i in ids):
            return True

    if element.get("title", "").strip():
        return True

    if element.get("alt", "").strip():
        return True

    if visible_text(element):
        return True

    element_id = element.get("id")

    if element_id:
        label = soup.find(
            "label",
            attrs={"for": element_id},
        )

        if label and visible_text(label):
            return True

    parent_label = element.find_parent("label")

    if parent_label and visible_text(parent_label):
        return True

    return False


def _contrast_ratio(foreground, background):
    """Return WCAG contrast ratio for two RGB colors."""
    def lum(rgb):
        vals = []
        for c in rgb:
            c = c / 255.0
            vals.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]
    a, b = lum(foreground), lum(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _parse_css_color(value):
    """Parse common CSS hex/rgb/named colors used in inline styles."""
    if not value:
        return None
    v = str(value).strip().lower()
    named = {
        'white': (255,255,255), 'black': (0,0,0), 'red': (255,0,0),
        'blue': (0,0,255), 'green': (0,128,0), 'gray': (128,128,128),
        'grey': (128,128,128), 'yellow': (255,255,0), 'orange': (255,165,0),
        'purple': (128,0,128), 'navy': (0,0,128), 'transparent': None,
    }
    if v in named:
        return named[v]
    m = re.fullmatch(r'#([0-9a-f]{3}|[0-9a-f]{6})', v)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = ''.join(c*2 for c in h)
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))
    m = re.fullmatch(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*[\d.]+)?\s*\)', v)
    if m:
        return tuple(int(m.group(i)) for i in (1,2,3))
    return None


def _extract_style_color(style, property_name):
    if not style:
        return None
    m = re.search(rf'{re.escape(property_name)}\s*:\s*([^;]+)', style, re.I)
    return _parse_css_color(m.group(1)) if m else None


def _specific_issue_details(issue):
    """Turn a generic rule finding into presentation-ready, concrete evidence."""
    raw = html.unescape(issue.get('element', ''))
    soup = BeautifulSoup(raw, 'lxml')
    el = soup.find()
    tag = el.name if el else 'element'
    attrs = dict(el.attrs) if el else {}
    text = visible_text(el) if el else ''
    text = re.sub(r'\s+', ' ', text).strip()
    rule = issue.get('rule', '')

    details = {
        'detected_value': '',
        'why_it_matters': 'This can create a barrier for people using screen readers, keyboard navigation, magnification or other assistive technology.',
        'specific_fix': issue.get('description', ''),
    }
    if rule == 'image-alt':
        details['detected_value'] = 'alt=' + repr(attrs.get('alt'))
        details['why_it_matters'] = 'A screen reader may announce the image as an unlabeled graphic, so a blind user may miss information conveyed visually.'
        details['specific_fix'] = 'If informative, replace the missing/empty alt with a concise description of the image. If purely decorative, keep alt="" intentionally.'
    elif rule == 'button-name':
        details['detected_value'] = f'<button> text={text!r}, aria-label={attrs.get("aria-label")!r}'
        details['why_it_matters'] = 'A screen-reader user can reach the control but may hear only “button” without knowing what action it performs.'
        details['specific_fix'] = 'Give the button a visible text label, or an aria-label that states the action (for example, “Close menu”).'
    elif rule == 'link-name':
        details['detected_value'] = f'<a href={attrs.get("href")!r}> accessible text={text!r}'
        details['why_it_matters'] = 'A screen-reader user navigating by links cannot determine where this link goes.'
        details['specific_fix'] = 'Use concise, meaningful link text such as “Download annual report” instead of an empty or icon-only link.'
    elif rule == 'label':
        details['detected_value'] = f'{tag} id={attrs.get("id")!r}, name={attrs.get("name")!r}, placeholder={attrs.get("placeholder")!r}'
        details['why_it_matters'] = 'Without a programmatic label, a screen reader may announce a form field without telling the user what information to enter.'
        details['specific_fix'] = 'Add a <label for="..."> connected to the control id, or provide a correct accessible name with aria-label/aria-labelledby.'
    elif rule == 'html-has-lang':
        details['detected_value'] = f'<html lang={attrs.get("lang")!r}'
        details['why_it_matters'] = 'Screen readers use the page language to choose pronunciation rules and voice settings.'
        details['specific_fix'] = 'Set the correct language, for example <html lang="en"> or <html lang="ta">.'
    elif rule == 'document-title':
        details['detected_value'] = '<title> is missing or empty'
        details['why_it_matters'] = 'Users of screen readers and browser tabs need a clear page title to identify the current page.'
        details['specific_fix'] = 'Add a unique, descriptive <title>, such as <title>Admissions — ABC University</title>.'
    elif rule == 'heading-order':
        details['detected_value'] = f'Found heading <{tag}>{text}</{tag}>'
        details['why_it_matters'] = 'Screen-reader users often navigate by headings; skipped levels can make the document structure confusing.'
        details['specific_fix'] = issue.get('description', '') + ' Change the heading level to reflect the actual content hierarchy.'
    elif rule == 'frame-title':
        details['detected_value'] = f'iframe title={attrs.get("title")!r}'
        details['why_it_matters'] = 'An iframe is a separate browsing context; without a title, a screen-reader user cannot tell what it contains.'
        details['specific_fix'] = 'Add a short title describing the embedded content, e.g. title="Campus map".'
    elif rule == 'video-caption':
        details['detected_value'] = 'No <track kind="captions"> detected inside this <video>.'
        details['why_it_matters'] = 'People who are deaf or hard of hearing can miss spoken information without synchronized captions.'
        details['specific_fix'] = 'Add a WebVTT captions track with kind="captions" and the correct srclang.'
    elif rule == 'duplicate-id':
        details['detected_value'] = attrs.get('id', '')
        details['why_it_matters'] = 'Duplicate IDs can break label associations, ARIA references and scripted focus behavior.'
        details['specific_fix'] = f'Give each element a unique id. For example, change repeated id="{attrs.get("id", "value")}" to distinct IDs.'
    elif rule == 'aria-hidden-focus':
        details['detected_value'] = 'aria-hidden="true" contains a focusable descendant'
        details['why_it_matters'] = 'Keyboard users can focus something that screen-reader users are told does not exist, creating an inconsistent experience.'
        details['specific_fix'] = 'Remove aria-hidden from the interactive region, or ensure hidden content is genuinely non-focusable.'
    elif rule == 'tabindex':
        details['detected_value'] = f'tabindex={attrs.get("tabindex")!r}'
        details['why_it_matters'] = 'Positive tabindex values can force an unexpected keyboard order that differs from the visual/document order.'
        details['specific_fix'] = 'Remove the positive tabindex and rely on natural DOM order; use tabindex="0" only when needed.'
    elif rule == 'empty-table-header':
        details['detected_value'] = f'<th> text={text!r}'
        details['why_it_matters'] = 'A blank table header makes it harder for screen-reader users to understand row/column relationships.'
        details['specific_fix'] = 'Give the header a meaningful label such as “Price”, “Date”, or “Status”.'
    elif rule == 'no-autoplay-audio':
        details['detected_value'] = '<' + tag + ' autoplay>'
        details['why_it_matters'] = 'Unexpected audio can interfere with screen-reader speech and can be disorienting for users with cognitive or sensory disabilities.'
        details['specific_fix'] = 'Remove autoplay or ensure users have an immediate, keyboard-accessible pause/mute control.'
    elif rule == 'contrast':
        details['detected_value'] = f'Foreground {issue.get("foreground")} on background {issue.get("background")} → contrast {issue.get("contrast_ratio")}:1'
        details['why_it_matters'] = 'Low contrast can make text difficult or impossible to read for people with low vision or color-vision differences.'
        details['specific_fix'] = 'Increase the contrast between foreground and background. WCAG AA generally requires 4.5:1 for normal text and 3:1 for large text.'
    issue.update(details)
    return issue


def add_contrast_issues(soup, issues):
    """Detect low-contrast pairs when explicit inline colors are available."""
    seen = set()
    for el in soup.find_all(style=True):
        style = el.get('style', '')
        fg = _extract_style_color(style, 'color')
        bg = _extract_style_color(style, 'background-color') or _extract_style_color(style, 'background')
        if not fg or not bg:
            continue
        ratio = _contrast_ratio(fg, bg)
        text = re.sub(r'\s+', ' ', visible_text(el)).strip()
        if not text or ratio >= 4.5:
            continue
        key = (str(el)[:500], round(ratio, 2))
        if key in seen:
            continue
        seen.add(key)
        severity = 5 if ratio < 3 else 4
        issue = {
            'rule': 'contrast', 'severity': severity,
            'title': 'Text has insufficient color contrast',
            'description': f'This element explicitly sets {fg} text on {bg} background, producing a {ratio:.2f}:1 contrast ratio.',
            'element': str(el)[:1500], 'wcag': 'WCAG 1.4.3 Contrast (Minimum)',
            'category': 'Visual Contrast', 'foreground': fg, 'background': bg,
            'contrast_ratio': round(ratio, 2),
        }
        issues.append(_specific_issue_details(issue))


def add_issue(
    issues,
    rule,
    severity,
    title,
    description,
    element,
    wcag="",
    category="Syntax",
):
    issue = {
        "rule": rule,
        "severity": int(severity),
        "title": title,
        "description": description,
        "element": str(element)[:1500],
        "wcag": wcag,
        "category": category,
    }
    issues.append(_specific_issue_details(issue))


# ============================================================
# RULE-BASED ACCESSIBILITY SCANNER
# ============================================================

def scan_page(html_text):
    soup = BeautifulSoup(html_text, "lxml")

    issues = []

    # --------------------------------------------------------
    # 1. Images
    # --------------------------------------------------------

    for img in soup.find_all("img"):

        alt = img.get("alt")

        if alt is None:
            add_issue(
                issues,
                "image-alt",
                5,
                "Image is missing alternative text",
                (
                    "A screen reader may not know what this image "
                    "represents. Add meaningful alt text, or use "
                    'alt="" when the image is purely decorative.'
                ),
                img,
                "WCAG 1.1.1 Non-text Content",
            )

        elif alt.strip() == "" and not (
            img.get("role") == "presentation"
            or img.get("aria-hidden") == "true"
        ):
            # Empty alt can be valid for decorative images, so make
            # this a lower-severity warning.
            add_issue(
                issues,
                "image-alt",
                3,
                "Image has empty alternative text",
                (
                    "Check whether this image is decorative. If it "
                    "conveys information, provide a meaningful alt."
                ),
                img,
                "WCAG 1.1.1 Non-text Content",
            )

    # --------------------------------------------------------
    # 2. Input type=image
    # --------------------------------------------------------

    for inp in soup.find_all(
        "input",
        attrs={"type": re.compile("^image$", re.I)},
    ):
        if not has_accessible_name(inp, soup):
            add_issue(
                issues,
                "input-image-alt",
                5,
                "Image input has no accessible name",
                "Add an alt attribute or another accessible name.",
                inp,
                "WCAG 1.1.1 Non-text Content",
            )

    # --------------------------------------------------------
    # 3. Buttons
    # --------------------------------------------------------

    for button in soup.find_all("button"):

        if not has_accessible_name(button, soup):

            add_issue(
                issues,
                "button-name",
                5,
                "Button has no accessible name",
                (
                    "Screen-reader users need to know what the "
                    "button does. Add visible text or aria-label."
                ),
                button,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 4. Links
    # --------------------------------------------------------

    for link in soup.find_all("a"):

        href = link.get("href")

        if href is not None and not has_accessible_name(link, soup):

            add_issue(
                issues,
                "link-name",
                5,
                "Link has no discernible text",
                (
                    "Add meaningful link text or an accessible "
                    "aria-label describing the destination."
                ),
                link,
                "WCAG 2.4.4 Link Purpose",
            )

    # --------------------------------------------------------
    # 5. Form controls
    # --------------------------------------------------------

    form_controls = soup.find_all(
        ["input", "textarea", "select"]
    )

    for control in form_controls:

        input_type = (
            control.get("type", "").lower()
        )

        if input_type in [
            "hidden",
            "submit",
            "button",
            "reset",
        ]:
            continue

        if not has_accessible_name(control, soup):

            add_issue(
                issues,
                "label",
                5,
                "Form control has no accessible label",
                (
                    "Associate this form control with a label "
                    "or provide an appropriate accessible name."
                ),
                control,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 6. HTML language
    # --------------------------------------------------------

    html_tag = soup.find("html")

    if html_tag:

        lang = html_tag.get("lang", "").strip()

        if not lang:

            add_issue(
                issues,
                "html-has-lang",
                4,
                "Page does not specify a language",
                (
                    'Add a language attribute such as '
                    '<html lang="en">.'
                ),
                html_tag,
                "WCAG 3.1.1 Language of Page",
            )

    # --------------------------------------------------------
    # 7. Page title
    # --------------------------------------------------------

    title = soup.find("title")

    if not title or not visible_text(title):

        add_issue(
            issues,
            "document-title",
            4,
            "Page is missing a meaningful title",
            (
                "Add a descriptive <title> so screen-reader and "
                "browser users can identify the page."
            ),
            title if title else soup,
            "WCAG 2.4.2 Page Titled",
        )

    # --------------------------------------------------------
    # 8. Main landmark
    # --------------------------------------------------------

    if not soup.find("main"):

        # ARIA main is also accepted.
        main_role = soup.find(
            attrs={"role": "main"}
        )

        if not main_role:

            add_issue(
                issues,
                "landmark",
                3,
                "No main content landmark detected",
                (
                    "Consider wrapping the primary content in "
                    "<main> so assistive technologies can navigate "
                    "the page structure more efficiently."
                ),
                soup.body if soup.body else soup,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 9. Navigation landmark
    # --------------------------------------------------------

    if not soup.find("nav") and not soup.find(
        attrs={"role": "navigation"}
    ):

        add_issue(
            issues,
            "navigation",
            2,
            "No navigation landmark detected",
            (
                "Consider using <nav> for major navigation regions."
            ),
            soup.body if soup.body else soup,
            "WCAG 1.3.1 Info and Relationships",
        )

    # --------------------------------------------------------
    # 10. Iframes
    # --------------------------------------------------------

    for frame in soup.find_all("iframe"):

        if not has_accessible_name(frame, soup):

            add_issue(
                issues,
                "frame-title",
                4,
                "Iframe has no accessible title",
                (
                    "Add a descriptive title attribute explaining "
                    "the iframe's content."
                ),
                frame,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 11. Video captions
    # --------------------------------------------------------

    for video in soup.find_all("video"):

        tracks = video.find_all("track")

        has_captions = any(
            str(t.get("kind", "")).lower() == "captions"
            for t in tracks
        )

        if not has_captions:

            add_issue(
                issues,
                "video-caption",
                5,
                "Video has no detected captions track",
                (
                    "Provide captions for spoken dialogue and "
                    "important audio information."
                ),
                video,
                "WCAG 1.2.2 Captions",
            )

    # --------------------------------------------------------
    # 12. Duplicate IDs
    # --------------------------------------------------------

    ids = {}

    for element in soup.find_all(attrs={"id": True}):

        element_id = element.get("id")

        if not element_id:
            continue

        ids.setdefault(element_id, []).append(element)

    for duplicate_id, elements in ids.items():

        if len(elements) > 1:

            add_issue(
                issues,
                "duplicate-id",
                4,
                "Duplicate HTML id detected",
                (
                    f'The id "{duplicate_id}" appears '
                    f"{len(elements)} times. IDs should be unique."
                ),
                elements[0],
                "WCAG 4.1.1 Parsing",
            )

    # --------------------------------------------------------
    # 13. aria-hidden + focusable content
    # --------------------------------------------------------

    for hidden in soup.find_all(
        attrs={"aria-hidden": "true"}
    ):

        focusable = hidden.find(
            [
                "a",
                "button",
                "input",
                "select",
                "textarea",
                "iframe",
            ]
        )

        tabindex = hidden.get("tabindex")

        if focusable is not None or (
            tabindex is not None
            and tabindex not in ["-1", ""]
        ):

            add_issue(
                issues,
                "aria-hidden-focus",
                5,
                "Hidden content contains focusable content",
                (
                    'Do not place focusable controls inside an '
                    'aria-hidden="true" region.'
                ),
                hidden,
                "WCAG 4.1.2 Name, Role, Value",
            )

    # --------------------------------------------------------
    # 14. Positive tabindex
    # --------------------------------------------------------

    for element in soup.find_all(
        attrs={"tabindex": True}
    ):

        try:
            tabindex = int(element.get("tabindex"))
        except Exception:
            continue

        if tabindex > 0:

            add_issue(
                issues,
                "tabindex",
                3,
                "Positive tabindex detected",
                (
                    "Avoid positive tabindex values. Prefer the "
                    "natural document order and tabindex='0' only "
                    "when an element genuinely needs focus."
                ),
                element,
                "WCAG 2.4.3 Focus Order",
            )

    # --------------------------------------------------------
    # 15. Heading hierarchy
    # --------------------------------------------------------

    headings = soup.find_all(
        re.compile("^h[1-6]$")
    )

    previous_level = None

    for heading in headings:

        level = int(heading.name[1])

        if (
            previous_level is not None
            and level > previous_level + 1
        ):

            add_issue(
                issues,
                "heading-order",
                3,
                "Heading hierarchy skips a level",
                (
                    f"The page jumps from H{previous_level} "
                    f"to H{level}. Use a logical heading hierarchy."
                ),
                heading,
                "WCAG 1.3.1 Info and Relationships",
            )

        previous_level = level

    # --------------------------------------------------------
    # 16. Empty table headers
    # --------------------------------------------------------

    for th in soup.find_all("th"):

        if not has_accessible_name(th, soup):

            add_issue(
                issues,
                "empty-table-header",
                3,
                "Table header has no discernible text",
                (
                    "Give table headers meaningful text so screen "
                    "reader users can understand the table structure."
                ),
                th,
                "WCAG 1.3.1 Info and Relationships",
            )

    # --------------------------------------------------------
    # 17. Autoplay audio/video
    # --------------------------------------------------------

    for media in soup.find_all(
        ["audio", "video"]
    ):

        if media.has_attr("autoplay"):

            add_issue(
                issues,
                "no-autoplay-audio",
                3,
                "Media is configured to autoplay",
                (
                    "Avoid unexpected audio. Give users an obvious "
                    "mechanism to pause, stop or mute it."
                ),
                media,
                "WCAG 1.4.2 Audio Control",
            )

    # --------------------------------------------------------
    # 18. ARIA hidden body
    # --------------------------------------------------------

    if soup.body and soup.body.get("aria-hidden") == "true":

        add_issue(
            issues,
            "aria-hidden-body",
            5,
            "Document body is aria-hidden",
            (
                "The main document body should not normally be "
                'removed from the accessibility tree.'
            ),
            soup.body,
            "WCAG 4.1.2 Name, Role, Value",
        )

    # --------------------------------------------------------
    # 19. Color contrast (explicit inline CSS heuristic)
    # --------------------------------------------------------
    add_contrast_issues(soup, issues)

    # --------------------------------------------------------
    # 20. Role=text heuristic
    # --------------------------------------------------------

    for element in soup.find_all(
        attrs={"role": "text"}
    ):

        if element.find(
            [
                "button",
                "a",
                "input",
                "select",
                "textarea",
            ]
        ):

            add_issue(
                issues,
                "aria-text",
                4,
                "role=text is used around interactive content",
                (
                    "Review this ARIA usage and avoid masking "
                    "interactive descendants from assistive "
                    "technologies."
                ),
                element,
                "WCAG 4.1.2 Name, Role, Value",
            )

    return soup, issues


# ============================================================
# ML ENRICHMENT
# ============================================================

def enrich_issues_with_dataset(
    issues,
    df,
    model_vectorizer,
    impact_model,
    search_vectorizer,
    search_matrix,
):
    if not issues:
        return issues

    query_texts = []

    for issue in issues:
        query_texts.append(
            " ".join(
                [
                    issue["rule"],
                    issue["title"],
                    issue["description"],
                    issue["element"],
                    issue["wcag"],
                    issue["category"],
                ]
            )
        )

    # ML-predicted impact
    X_query = model_vectorizer.transform(query_texts)
    predicted = impact_model.predict(X_query)

    # Dataset semantic similarity
    Q = search_vectorizer.transform(query_texts)
    similarities = cosine_similarity(
        Q,
        search_matrix,
    )

    for i, issue in enumerate(issues):

        # Clamp to AccessGuru's normal 1-5 scale
        ml_score = float(
            np.clip(predicted[i], 1.0, 5.0)
        )

        best_indices = np.argsort(
            similarities[i]
        )[-3:][::-1]

        best_idx = int(best_indices[0])

        dataset_row = df.iloc[best_idx]

        issue["ml_impact"] = round(
            ml_score,
            2,
        )

        issue["dataset_similarity"] = round(
            float(similarities[i][best_idx]),
            3,
        )

        issue["dataset_rule"] = str(
            dataset_row.get(
                "violationnumberID",
                "",
            )
        )

        issue["dataset_description"] = str(
            dataset_row.get(
                "Description",
                "",
            )
        )

        issue["dataset_category"] = str(
            dataset_row.get(
                "category",
                "",
            )
        )

        issue["dataset_fix"] = str(
            dataset_row.get(
                "fixed_code_desc",
                "",
            )
        )

        issue["dataset_wcag"] = str(
            dataset_row.get(
                "updated_Description",
                "",
            )
        )

        issue["dataset_source_url"] = str(
            dataset_row.get(
                "webURL",
                "",
            )
        )

    return issues


# ============================================================
# SCORE
# ============================================================

def calculate_accessibility_score(issues, html_text):
    """
    Explainable 0-100 WCAG-oriented screening score.

    The score is calculated from the automated rule checks themselves.
    AccessGuru/ML impact is NOT used to create the percentage.

    Each applicable rule has a fixed maximum weight. A finding loses part of
    that rule's weight according to how many applicable elements fail and the
    scanner's fixed severity. The function also returns a breakdown so the UI
    can show exactly where points were lost.
    """
    soup = BeautifulSoup(html_text, "lxml")

    weights = {
        "image-alt": 15.0,
        "input-image-alt": 3.0,
        "button-name": 10.0,
        "link-name": 10.0,
        "label": 15.0,
        "html-has-lang": 5.0,
        "document-title": 5.0,
        "landmark": 4.0,
        "navigation": 2.0,
        "frame-title": 4.0,
        "video-caption": 7.0,
        "duplicate-id": 5.0,
        "aria-hidden-focus": 5.0,
        "tabindex": 3.0,
        "heading-order": 4.0,
        "empty-table-header": 3.0,
        "no-autoplay-audio": 3.0,
        "aria-hidden-body": 2.0,
        "contrast": 10.0,
        "aria-text": 3.0,
    }

    page_level_rules = {
        "html-has-lang", "document-title", "landmark", "navigation",
        "duplicate-id", "aria-hidden-body"
    }

    opportunities = {
        "image-alt": len(soup.find_all("img")),
        "input-image-alt": len(soup.find_all("input", attrs={"type": re.compile("^image$", re.I)})),
        "button-name": len(soup.find_all("button")),
        "link-name": sum(1 for x in soup.find_all("a") if x.get("href") is not None),
        "label": sum(
            1 for x in soup.find_all(["input", "textarea", "select"])
            if x.get("type", "").lower() not in ["hidden", "submit", "button", "reset"]
        ),
        "html-has-lang": 1 if soup.find("html") else 0,
        "document-title": 1,
        "landmark": 1,
        "navigation": 1,
        "frame-title": len(soup.find_all("iframe")),
        "video-caption": len(soup.find_all("video")),
        "duplicate-id": 1 if soup.find_all(attrs={"id": True}) else 0,
        "aria-hidden-focus": len(soup.find_all(attrs={"aria-hidden": "true"})),
        "tabindex": len(soup.find_all(attrs={"tabindex": True})),
        "heading-order": max(0, len(soup.find_all(re.compile("^h[1-6]$"))) - 1),
        "empty-table-header": len(soup.find_all("th")),
        "no-autoplay-audio": len(soup.find_all(["audio", "video"])),
        "aria-hidden-body": 1 if soup.body else 0,
        "contrast": sum(
            1 for x in soup.find_all(style=True)
            if _extract_style_color(x.get("style", ""), "color")
            and (_extract_style_color(x.get("style", ""), "background-color")
                 or _extract_style_color(x.get("style", ""), "background"))
            and visible_text(x)
        ),
        "aria-text": len(soup.find_all(attrs={"role": "text"})),
    }

    applicable_weight = 0.0
    total_loss = 0.0
    breakdown = []

    for rule, weight in weights.items():
        opportunity_count = opportunities.get(rule, 0)
        if opportunity_count <= 0:
            continue

        applicable_weight += weight
        rule_issues = [x for x in issues if x.get("rule") == rule]
        failed_count = len(rule_issues)

        if not rule_issues:
            continue

        if rule in page_level_rules:
            failed_ratio = 1.0
        else:
            failed_ratio = min(1.0, failed_count / float(opportunity_count))

        severity_factor = sum(
            min(5.0, max(1.0, float(x.get("severity", 1)))) / 5.0
            for x in rule_issues
        ) / max(1, failed_count)

        raw_rule_loss = weight * failed_ratio * severity_factor
        score_loss = (100.0 * raw_rule_loss / max(weight, 0.0001))
        # score_loss is the percentage of the rule's available contribution.
        # Normalize it later against all applicable rule weights.
        normalized_loss = 100.0 * raw_rule_loss
        total_loss += normalized_loss

        title = str(rule_issues[0].get("title", rule.replace("-", " ").title()))
        wcag = str(rule_issues[0].get("wcag", ""))
        severity_value = max(float(x.get("severity", 1)) for x in rule_issues)
        severity_text = severity_name(severity_value)

        breakdown.append({
            "rule": rule,
            "title": title,
            "wcag": wcag,
            "affected": failed_count,
            "opportunities": opportunity_count,
            "severity": severity_text,
            "weight": weight,
            "raw_loss": raw_rule_loss,
            "severity_factor": severity_factor,
        })

    if applicable_weight <= 0:
        return 100.0, []

    # Convert weighted rule losses into a 0-100 score.
    score = 100.0 - (100.0 * sum(x["raw_loss"] for x in breakdown) / applicable_weight)
    score = round(max(0.0, min(100.0, score)), 1)

    # Store the actual percentage-point contribution of every finding group.
    for item in breakdown:
        item["points_lost"] = round(
            100.0 * item["raw_loss"] / applicable_weight, 1
        )

    breakdown.sort(key=lambda x: x["points_lost"], reverse=True)
    return score, breakdown


def score_label(score):
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 65:
        return "Needs Improvement"
    if score >= 50:
        return "Poor"
    return "Critical"


def severity_name(score):
    score = float(score)

    if score >= 4.5:
        return "Critical"
    if score >= 3.5:
        return "Serious"
    if score >= 2.5:
        return "Moderate"
    return "Minor"


# ============================================================
# REPORT
# ============================================================

def make_report(url, score, issues, features):
    lines = []

    lines.append("ACCESSIFY — AUTOMATED ACCESSIBILITY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"URL: {url}")
    lines.append(f"Accessibility Screening Score: {score}/100")
    lines.append(f"Rating: {score_label(score)}")
    lines.append("")
    lines.append("NOTE:")
    lines.append(
        "This is an automated screening estimate, not a WCAG "
        "conformance certification."
    )
    lines.append("")
    lines.append("PAGE SUMMARY")
    lines.append("-" * 60)

    for key, value in features.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("DETECTED ISSUES")
    lines.append("-" * 60)

    if not issues:
        lines.append("No automated issues detected.")

    for i, issue in enumerate(
        sorted(
            issues,
            key=lambda x: -float(
                x.get(
                    "ml_impact",
                    x["severity"],
                )
            ),
        ),
        1,
    ):

        impact = issue.get(
            "ml_impact",
            issue["severity"],
        )

        lines.append(
            f"{i}. {issue['title']} "
            f"[{severity_name(impact)}]"
        )

        lines.append(
            f"   Rule: {issue['rule']}"
        )

        lines.append(
            f"   WCAG: {issue['wcag']}"
        )

        lines.append(f"   What is wrong: {issue.get('description', '')}")
        lines.append(f"   Detected value: {issue.get('detected_value', '')}")
        lines.append(f"   Why it matters: {issue.get('why_it_matters', '')}")
        lines.append(f"   Specific fix: {issue.get('specific_fix', '')}")
        lines.append(f"   Element: {issue['element']}")

        if issue.get("dataset_fix"):
            lines.append(
                "   Dataset-grounded recommendation:"
            )
            lines.append(
                "   "
                + issue["dataset_fix"][:2000]
                .replace("\n", " ")
            )

        lines.append("")

    return "\n".join(lines)


def find_source_line_for_issue(issue, source_html):
    """Best-effort source line lookup for the offending HTML element."""
    try:
        fragment = BeautifulSoup(str(issue.get("element", "")), "html.parser").find()
    except Exception:
        fragment = None
    if fragment is None:
        return None, ""

    tag_name = str(fragment.name or "").lower()
    attrs = fragment.attrs or {}
    lines = str(source_html or "").splitlines()
    if not lines:
        return None, ""

    # Prioritise unique attributes to avoid matching the wrong repeated tag.
    keys = ["id", "name", "src", "href", "for", "aria-label", "alt", "type", "role"]
    needles = []
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        if value is not None and str(value).strip():
            escaped = html.escape(str(value), quote=True)
            needles.append((key.lower(), str(value).strip(), escaped))
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        if tag_name and f"<{tag_name}" not in low:
            continue
        for key, value, escaped in needles:
            if re.search(r"\b" + re.escape(key) + r"\s*=\s*(['\"])" + re.escape(value) + r"\1", line, re.I) or escaped.lower() in low:
                return idx, line.strip()[:260]
    # Fallback: locate the first line containing this tag name.
    for idx, line in enumerate(lines, start=1):
        if tag_name and re.search(r"<\s*" + re.escape(tag_name) + r"\b", line, re.I):
            return idx, line.strip()[:260]
    return None, ""


# ============================================================
# UI HEADER
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-topline"><span class="hero-dot"></span> AI-ASSISTED WEB ACCESSIBILITY</div>'
    '<div class="brand">♿ ACCESS<span class="brand-mark">IFY</span></div>'
    '<div class="project-name">Make every click count. Make the web work for everyone.</div>'
    '<div class="hero-desc">An automated auditing tool that scans websites for accessibility flaws—such as missing structural tags, unclear controls, and poor visual contrast—and gives developers practical ways to improve the experience for everyone.</div>'
    '<div class="hero-pills"><span class="hero-pill">✦ Smart issue detection</span><span class="hero-pill">◈ WCAG-oriented checks</span><span class="hero-pill">⌁ Explainable risk insights</span></div>'
    '<div class="hero-float"><div class="hero-float-label">Built for inclusion</div><div class="hero-float-number">Access for all.</div><div class="hero-float-sub">Better access. Better experiences.</div></div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FIRST-SCREEN URL SCAN PANEL
# ============================================================

st.markdown(
    '<section class="scan-intro">'
    '<div class="scan-kicker"><span class="scan-kicker-dot"></span> READY WHEN YOU ARE</div>'
    '<div class="scan-title">Give your website an <span>accessibility check.</span></div>'
    '<div class="scan-copy">Paste a public URL. Accessify will inspect the page, identify potential barriers, and turn findings into practical next steps.</div>'
    '</section>',
    unsafe_allow_html=True,
)

with st.container(key="scan_input_panel"):
    with st.form("accessify_scan_form", clear_on_submit=False):
        url_input = st.text_input(
            "Website URL",
            placeholder="https://your-website.com",
            label_visibility="collapsed",
        )
        form_cols = st.columns([3, 1])
        with form_cols[0]:
            analyze = st.form_submit_button(
                "✦  Analyze website",
                type="primary",
                use_container_width=True,
            )
        with form_cols[1]:
            st.markdown('<div class="enter-hint">PRESS ENTER ↵<br><span>or click Analyze</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="scan-footnote"><span>◈</span> Public websites only &nbsp;·&nbsp; WCAG-oriented checks &nbsp;·&nbsp; Dataset-assisted insights</div>',
    unsafe_allow_html=True,
)

# Official WCAG reference
st.markdown(
    '<div style="margin:-0.5rem 0 1rem 0;">'
    '<a href="https://www.w3.org/WAI/standards-guidelines/wcag/" target="_blank" style="text-decoration:none;">'
    '<button style="padding:0.6rem 1rem;border-radius:10px;border:1px solid #6366f1;background:#eef2ff;color:#312e81;font-weight:800;cursor:pointer;">📖 Learn about WCAG rules ↗</button>'
    '</a></div>',
    unsafe_allow_html=True,
)




# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("♿ Accessify")

    st.write(
        """
        Accessify helps developers discover accessibility
        barriers that can affect people using screen readers,
        keyboard navigation and other assistive technologies.
        """
    )

    st.divider()

    st.subheader("Jump to a feature")
    st.markdown(
        """
        <div class="side-nav">
          <a href="#detected-problems">🚨 Detected Problems</a>
          <a href="#developer-recommendations">🛠 Developer Recommendations</a>
          <a href="#junk-alt-text">🖼️ Junk Alt-Text Detector — NLP Classifier</a>
          <a href="#risk-predictor">🚦 Accessibility Risk</a>
          <a href="#explainable-ai">📊 Code-Line Error Likelihood</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Technology")

    st.write(
        """
        • Python + BeautifulSoup  
        • Scikit-learn ML  
        • TF-IDF semantic matching  
        • AccessGuru accessibility dataset  
        • WCAG-oriented rule checks  
        • Streamlit dashboard
        """
    )

    st.divider()

    st.subheader("Important")

    st.caption(
        "Automated accessibility testing cannot prove that a "
        "website is fully accessible. Manual testing and testing "
        "with people who use assistive technologies are still important."
    )


# ============================================================
# DATASET INITIALIZATION
# ============================================================

dataset_path = find_dataset()

if not dataset_path:

    st.error(
        f"Could not find `{DATASET_NAME}`."
    )

    st.info(
        "Place Original_full_data.csv in the same folder as app.py "
        "and restart Streamlit."
    )

    st.stop()

try:

    with st.spinner(
        "Loading AccessGuru accessibility dataset..."
    ):

        dataset = load_accessguru_dataset(
            dataset_path
        )

        model_vectorizer, impact_model = (
            train_accessguru_model(dataset)
        )

        search_vectorizer, search_matrix = (
            build_dataset_search_index(dataset)
        )

        risk_model, risk_category_frequency = train_tabular_risk_model(dataset)

except Exception as e:

    st.error(
        "The dataset could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DATASET INFO
# ============================================================

with st.expander(
    "📊 Dataset used by the AI",
    expanded=False,
):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(dataset):,}",
    )

    c2.metric(
        "Violation Types",
        dataset["violationnumberID"].nunique(),
    )

    c3.metric(
        "Categories",
        dataset["category"].nunique(),
    )

    c4.metric(
        "Domains",
        dataset["domain"].nunique()
        if "domain" in dataset.columns
        else "—",
    )

    st.caption(
        "The uploaded AccessGuru dataset is actively used by the ML layer: "
        "Accessify trains a TF-IDF + Ridge impact model and performs semantic retrieval against real-world accessibility violation records."
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    url = normalize_url(url_input)

    if not url:

        st.warning(
            "Please enter a valid website URL."
        )

        st.stop()

    try:

        with st.spinner(
            "Fetching webpage..."
        ):

            page_html, final_url = fetch_page(url)

        with st.spinner(
            "Scanning accessibility barriers..."
        ):

            soup, issues = scan_page(
                page_html
            )

        with st.spinner(
            "Applying AccessGuru ML model..."
        ):

            issues = enrich_issues_with_dataset(
                issues,
                dataset,
                model_vectorizer,
                impact_model,
                search_vectorizer,
                search_matrix,
            )

        score, score_breakdown = calculate_accessibility_score(
            issues,
            page_html,
        )

        # ----------------------------------------------------
        # Summary features
        # ----------------------------------------------------

        images = soup.find_all("img")

        buttons = soup.find_all("button")

        inputs = soup.find_all(
            ["input", "textarea", "select"]
        )

        links = soup.find_all("a")

        summary = {
            "Images": len(images),
            "Buttons": len(buttons),
            "Form controls": len(inputs),
            "Links": len(links),
            "Detected issues": len(issues),
        }

        # Store in session state for download / rerender.
        st.session_state["analysis"] = {
            "url": final_url,
            "score": score,
            "score_breakdown": score_breakdown,
            "issues": issues,
            "page_html": page_html,
            "summary": summary,
            "report": make_report(
                final_url,
                score,
                issues,
                summary,
            ),
        }

    except requests.exceptions.Timeout:

        st.error(
            "The website took too long to respond."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        st.error(
            "Could not access the website."
        )

        st.code(str(e))

        st.stop()

    except Exception as e:

        st.error(
            "Something went wrong during analysis."
        )

        st.exception(e)

        st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    score = result["score"]
    score_breakdown = result.get("score_breakdown", [])
    issues = result["issues"]
    summary = result["summary"]

    st.divider()

    st.markdown("## 📈 Accessibility Results")

    st.markdown(
        f'<div class="error-badge">⚠️ {len(issues)} detected accessibility issue(s)</div>',
        unsafe_allow_html=True,
    )

    score_col, stats_col = st.columns(
        [1, 2]
    )

    with score_col:

        st.markdown(
            f"""
            <div class="score-wrap">
                <div style="font-weight:800;opacity:.8">ACCESSIFY ACCESSIBILITY SCORE</div>
                <div class="score-number">{score}</div>
                <div class="score-label">{score_label(score)}</div>
                <div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>
                <div style="margin-top:.7rem;font-size:.82rem;opacity:.78">100 = strongest automated result</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("WCAG-oriented automated screening score — not an official WCAG conformance percentage or certification.")

        # ----------------------------------------------------
        # EXPLAINABLE SCORE BREAKDOWN
        # ----------------------------------------------------
        st.markdown("### 🧮 Why did points get lost?")
        st.caption("Accessify starts at 100.0. The rows below show exactly which automated checks caused the score to fall.")

        if not score_breakdown:
            st.success("No applicable automated accessibility failures were found, so no points were deducted.")
        else:
            displayed_loss = 0.0
            for row_start in range(0, len(score_breakdown), 2):
                deduction_cols = st.columns(2, gap="medium")
                for col, item in zip(deduction_cols, score_breakdown[row_start:row_start + 2]):
                    loss = float(item["points_lost"])
                    displayed_loss += loss
                    wcag_text = f" · {item['wcag']}" if item["wcag"] else ""
                    affected_text = f"{item['affected']} affected"
                    with col:
                        st.markdown(
                            f'<div class="deduction-card">'
                            f'<div style="color:#E10600;font-weight:900;font-size:1.05rem">−{loss:.1f} points</div>'
                            f'<div style="font-weight:800;margin:.22rem 0">{html.escape(str(item["title"]))}</div>'
                            f'<div style="font-size:.88rem;opacity:.78">{affected_text} of {item["opportunities"]} opportunities · {html.escape(str(item["severity"]))}{html.escape(wcag_text)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown(
                f"**Score calculation: 100.0 − {displayed_loss:.1f} = {score:.1f}**"
            )

            with st.expander("📘 How Accessify calculates the score"):
                st.markdown(
                    """
                    **1. Start at 100 points.**

                    **2. Only applicable checks count.** A rule such as video captions is not counted when the page contains no video.

                    **3. Each WCAG-oriented rule has a fixed weight.** More important automated checks contribute more to the score than minor checks.

                    **4. Repeated failures are counted by affected elements.** One missing image alternative is not treated the same as dozens of missing alternatives.

                    **5. Scanner severity adjusts the loss.** The fixed rule severity is used for the score; the AccessGuru/ML impact prediction is kept separate for prioritization.

                    **6. The final score is normalized to 0–100.** Every deduction shown above is a percentage-point contribution to the displayed score.

                    This is an **automated WCAG-oriented screening score**, not a claim that a website is officially WCAG conformant.
                    """
                )

    with stats_col:

        cols = st.columns(5)

        for i, (key, value) in enumerate(
            summary.items()
        ):

            cols[i].metric(
                key,
                value,
            )

    # --------------------------------------------------------
    # Overall message
    # --------------------------------------------------------

    if score >= 90:

        st.success(
            "🎉 The automated scanner found relatively few "
            "accessibility barriers."
        )

    elif score >= 75:

        st.info(
            "👍 The page has a reasonable accessibility baseline, "
            "but some barriers should still be fixed."
        )

    elif score >= 60:

        st.warning(
            "⚠️ Several accessibility barriers were detected. "
            "Prioritize the serious and critical findings."
        )

    else:

        st.error(
            "🚨 Significant accessibility barriers were detected. "
            "The page should undergo accessibility remediation."
        )

    # --------------------------------------------------------
    # Issue distribution
    # --------------------------------------------------------

    st.markdown('<div id="detected-problems" class="section-anchor"><h2>🚨 Detected Problems</h2></div>', unsafe_allow_html=True)

    if not issues:

        st.success(
            "No automated accessibility issues were detected."
        )

    else:

        # Add best-effort line numbers and the matching returned-HTML source line
        # directly to the main Detected Problems table.
        source_html_for_table = result.get("page_html", "")
        issue_rows = []
        for x in issues:
            source_line, source_code = find_source_line_for_issue(x, source_html_for_table)
            issue_rows.append({
                "What is wrong": x.get("detected_value") or x["title"],
                "Line number": source_line if source_line is not None else "Not resolved",
                "Code at line": source_code or re.sub(r"\s+", " ", html.unescape(x.get("element", "")))[:260] or "Source snippet unavailable",
                "Evidence / element": re.sub(
                    r"\s+",
                    " ",
                    html.unescape(x.get("element", "")),
                )[:220],
                "Why it matters": x.get("why_it_matters", ""),
                "Specific fix": x.get("specific_fix", x.get("description", "")),
                "Severity": severity_name(x.get("ml_impact", x["severity"])),
                "ML impact": round(x.get("ml_impact", x["severity"]), 2),
            })
        issue_df = pd.DataFrame(issue_rows)

        issue_df = issue_df.sort_values(
            "ML impact",
            ascending=False,
        )

        st.dataframe(
            issue_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Line numbers refer to the fetched HTML response when a matching source line can be identified. "
            "If the page is minified, generated by JavaScript, or contains repeated elements, the line may be unavailable or approximate."
        )

        # ----------------------------------------------------
        # Detailed recommendations
        # ----------------------------------------------------

        st.markdown(
            '<div id="developer-recommendations" class="section-anchor"><h2>🛠 Developer Recommendations</h2></div>',
            unsafe_allow_html=True,
        )

        sorted_issues = sorted(
            issues,
            key=lambda x: -float(
                x.get(
                    "ml_impact",
                    x["severity"],
                )
            ),
        )

        for number, issue in enumerate(
            sorted_issues,
            1,
        ):

            impact = issue.get(
                "ml_impact",
                issue["severity"],
            )

            severity = severity_name(
                impact
            )

            with st.expander(
                f"{number}. {issue['title']} "
                f"— {severity}"
            ):

                left, right = st.columns(
                    [1, 1]
                )

                with left:

                    st.markdown(
                        f"**Detected rule:** `{issue['rule']}`"
                    )

                    st.markdown(
                        f"**WCAG area:** "
                        f"{issue['wcag'] or 'General accessibility'}"
                    )

                    st.markdown(
                        f"**Category:** {issue['category']}"
                    )

                    st.markdown(
                        f"**ML-predicted impact:** "
                        f"`{impact:.2f}/5`"
                    )

                    if issue.get(
                        "dataset_rule"
                    ):

                        st.markdown(
                            "**Closest AccessGuru violation:** "
                            f"`{issue['dataset_rule']}`"
                        )

                        st.markdown(
                            f"Dataset similarity: "
                            f"`{issue.get('dataset_similarity', 0):.3f}`"
                        )

                with right:

                    st.markdown("**🔴 What exactly is wrong**")
                    st.write(issue.get("description", ""))
                    st.markdown("**Detected value / evidence**")
                    st.code(issue.get("detected_value", "Not available"), language="text")
                    st.markdown("**👁 Why this matters for accessibility**")
                    st.write(issue.get("why_it_matters", ""))

                    st.markdown(
                        "**Detected HTML**"
                    )

                    st.code(
                        html.unescape(
                            issue["element"]
                        ),
                        language="html",
                    )

                st.markdown("**🟢 Recommended fix for THIS finding**")
                st.info(issue.get("specific_fix", issue.get("description", "")))

                st.markdown("**Recommended fix / AccessGuru evidence**")

                dataset_fix = (
                    issue.get(
                        "dataset_fix",
                        "",
                    )
                    or ""
                )

                # Avoid showing huge generated text blocks
                # when the dataset contains them.
                if (
                    dataset_fix
                    and dataset_fix.lower() not in [
                        "nan",
                        "none",
                    ]
                ):

                    st.info(
                        "Recommendation retrieved from the "
                        "closest matching AccessGuru examples."
                    )

                    st.markdown(
                        dataset_fix[:5000]
                    )

                # Practical generic code examples
                rule = issue["rule"]

                if rule == "image-alt":

                    st.code(
                        """<!-- Informative image -->
<img src="doctor.jpg"
     alt="Doctor speaking with a patient">

<!-- Decorative image -->
<img src="decoration.svg"
     alt="">""",
                        language="html",
                    )

                elif rule == "button-name":

                    st.code(
                        """<!-- Good -->
<button aria-label="Close menu">
    <svg aria-hidden="true">...</svg>
</button>""",
                        language="html",
                    )

                elif rule == "label":

                    st.code(
                        """<label for="email">
    Email address
</label>

<input
    id="email"
    type="email"
    name="email">""",
                        language="html",
                    )

                elif rule == "html-has-lang":

                    st.code(
                        """<html lang="en">""",
                        language="html",
                    )

                elif rule == "frame-title":

                    st.code(
                        """<iframe
    src="..."
    title="Product demonstration video">
</iframe>""",
                        language="html",
                    )

                elif rule == "video-caption":

                    st.code(
                        """<video controls>
    <source src="video.mp4"
            type="video/mp4">

    <track
        kind="captions"
        src="captions.vtt"
        srclang="en"
        label="English">
</video>""",
                        language="html",
                    )

                elif rule == "heading-order":

                    st.code(
                        """<h1>Products</h1>
<h2>Computers</h2>
<h3>Laptops</h3>""",
                        language="html",
                    )

                elif rule == "duplicate-id":

                    st.code(
                        """<!-- IDs must be unique -->
<input id="email">
<input id="phone">""",
                        language="html",
                    )

                elif rule == "aria-hidden-focus":

                    st.code(
                        """<!-- Do not make aria-hidden
     content focusable -->

<div aria-hidden="true">
    <span>Decorative content</span>
</div>""",
                        language="html",
                    )

                elif rule == "contrast":

                    st.code(
                        """/* Too low */
color: #777;
background: #fff;

/* Better: choose a darker foreground */
color: #333;
background: #fff;""",
                        language="css",
                    )

    # --------------------------------------------------------
    # Download report
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "## 📄 Export"
    )

    report_bytes = result["report"].encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download Accessibility Report",
        data=report_bytes,
        file_name="accessify_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Methodology
    # --------------------------------------------------------

    with st.expander(
        "🧠 How Accessify works"
    ):

        st.markdown(
            """
            **Step 1 — Web analysis**

            The application downloads the submitted webpage and
            parses its HTML.

            **Step 2 — Accessibility checks**

            It checks practical accessibility signals such as
            alternative text, button names, form labels, headings,
            landmarks, language, iframe titles, captions, duplicate
            IDs, ARIA usage and keyboard-related attributes.

            **Step 3 — AccessGuru ML**

            The AccessGuru dataset contains real-world accessibility
            violation examples. The application trains a lightweight
            TF-IDF + Ridge model to estimate violation impact from
            violation descriptions and HTML.

            **Step 4 — Semantic retrieval**

            Each detected issue is compared with the AccessGuru
            dataset using TF-IDF cosine similarity. The closest
            known violation is used to provide additional context
            and a dataset-grounded remediation.

            **Step 5 — Screening score**

            The detected issues and their estimated severity are
            combined into an interpretable 0–100 screening score.

            **Step 6 — Developer action**

            Instead of only saying "your website failed", the
            application shows the offending HTML and practical code
            examples for fixing it.
            """
        )

    st.caption(
        "Accessify • Automated web accessibility screening"
    )

    # ========================================================
    # ADVANCED ACCESSIBILITY INTELLIGENCE (ADDED FEATURES)
    # Kept after the existing report and methodology sections.
    # ========================================================

    st.divider()
    st.markdown("## More Accessibility Insights")
    st.caption("Extra checks to spot misleading image text, estimate risk, and show what the model learned.")

    try:
        analysed_soup = BeautifulSoup(result.get("page_html", ""), "html.parser")
        junk_alt_findings = detect_junk_alt_text(analysed_soup)
    except Exception:
        junk_alt_findings = []

    st.markdown('<div id="junk-alt-text" class="feature-title">🖼️ Junk Alt-Text Detector — NLP Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Uses a TF-IDF + Logistic Regression NLP classifier to flag potentially unhelpful image descriptions for review.</div>', unsafe_allow_html=True)
    if junk_alt_findings:
        st.markdown(
            f'<div class="risk-panel"><strong>🔴 {len(junk_alt_findings)} likely junk alt-text finding(s)</strong></div>',
            unsafe_allow_html=True,
        )
        # Summary table makes the NLP classifier's output visible at a glance.
        nlp_table_rows = []
        for finding in junk_alt_findings:
            nlp_table_rows.append({
                "Alt text detected": finding.get("alt", ""),
                "NLP prediction": finding.get("nlp_prediction", "Review needed"),
                "Model confidence": f"{float(finding.get('nlp_confidence', 0)):.0%}",
                "Why flagged": "; ".join(finding.get("reasons", [])),
                "Image source": finding.get("src", "") or "No src attribute",
            })
        st.dataframe(pd.DataFrame(nlp_table_rows), use_container_width=True, hide_index=True)
        for i, finding in enumerate(junk_alt_findings, 1):
            with st.expander(f"{i}. Unhelpful alt text: {finding['alt'][:100]}"):
                st.markdown("**NLP classifier result**")
                st.write(f"Prediction: `{finding.get('nlp_prediction', 'not available')}`")
                st.write(f"Estimated model confidence for potentially unhelpful text: {finding.get('nlp_confidence', 0):.0%}")
                st.caption("This lightweight model uses curated example phrases for weak supervision; confidence is not a validated probability that the alt text is wrong.")
                st.markdown("**Why it was flagged**")
                for reason in finding["reasons"]:
                    st.write(f"• {reason}")
                st.markdown("**Image source**")
                st.code(finding["src"] or "No src attribute", language="text")
                st.markdown("**Detected HTML**")
                st.code(finding["element"], language="html")
                st.success(
                    "Suggested action: replace this with a concise description of the image's "
                    "purpose or content. If the image is purely decorative, use alt="" instead."
                )
    else:
        st.success(
            "No likely junk alt text was detected by these checks. This does not prove every "
            "alternative text is meaningful; review important images manually."
        )

    st.markdown('<div id="risk-predictor" class="feature-title">🚦 Accessibility Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Combines detected issue patterns to help prioritise what needs attention first.</div>', unsafe_allow_html=True)
    issue_count = len(issues)
    critical_count = sum(
        1 for issue in issues
        if float(issue.get("ml_impact", issue.get("severity", 1))) >= 4.5
    )
    serious_count = sum(
        1 for issue in issues
        if 3.5 <= float(issue.get("ml_impact", issue.get("severity", 1))) < 4.5
    )
    image_count = int(summary.get("Images", 0))
    controls_count = int(summary.get("Buttons", 0)) + int(summary.get("Form controls", 0))
    junk_count = len(junk_alt_findings)
    density = issue_count / max(image_count + controls_count + int(summary.get("Links", 0)), 1)

    risk_cols = st.columns(4)
    risk_cols[0].metric("Total issues", issue_count)
    risk_cols[1].metric("Critical / near-critical", critical_count)
    risk_cols[2].metric("Serious", serious_count)
    risk_cols[3].metric("Junk alt text", junk_count)

    # Use the AccessGuru-trained decision tree to classify each detected issue.
    issue_risk_rows = []
    for issue in issues:
        category = str(issue.get("category", "")).lower()
        feature_row = risk_text_features(
            issue.get("description", issue.get("title", "")),
            issue.get("element", ""),
            category,
            risk_category_frequency.get(category, 0),
        )
        predicted_class = str(risk_model.predict([feature_row])[0])
        issue_risk_rows.append({
            "Issue": issue.get("title", issue.get("rule", "Accessibility issue")),
            "Random Forest risk class": predicted_class,
            "Existing ML impact (/5)": round(float(issue.get("ml_impact", issue.get("severity", 1))), 2),
        })

    if issue_risk_rows:
        st.dataframe(pd.DataFrame(issue_risk_rows), use_container_width=True, hide_index=True)
        risk_classes = [row["Random Forest risk class"] for row in issue_risk_rows]
        if "Critical" in risk_classes or critical_count > 0:
            overall_risk = "HIGH"
            risk_color = "#E10600"
        elif "High" in risk_classes or serious_count > 0:
            overall_risk = "ELEVATED"
            risk_color = "#C2410C"
        elif issue_count > 0:
            overall_risk = "MODERATE"
            risk_color = "#B45309"
        else:
            overall_risk = "LOW"
            risk_color = "#15803D"
    else:
        overall_risk = "LOW"
        risk_color = "#15803D"
        st.write("No issues were detected to classify.")

    st.markdown(
        f'<div class="risk-panel"><span style="font-weight:800;color:{risk_color};font-size:1.1rem">Overall screening risk: {overall_risk}</span>'
        f'<br><span>Issue density: {density:.2f} detected issue(s) per image, control, or link counted. This is a prioritisation signal, not a formal compliance probability.</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The Decision Tree is trained on AccessGuru records, using text/HTML-derived tabular features "
        "to learn classes from the dataset's initial impact scores. Page-level risk also considers the "
        "observed issue counts and should be interpreted as a screening aid, not a certified risk score."
    )

    st.markdown('<div id="explainable-ai" class="feature-title">🧬 Accessibility Code-Line Error Likelihood</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-desc">Ranks the source lines connected to detected accessibility findings, so developers can see where to investigate first.</div>', unsafe_allow_html=True)
    source_html_for_lines = result.get("page_html", "")
    line_findings = []
    for issue in issues:
        line_number, source_fragment = find_source_line_for_issue(issue, source_html_for_lines)
        severity_value = float(issue.get("ml_impact", issue.get("severity", 1)))
        # This is a transparent heuristic indicator, not a calibrated statistical probability.
        likelihood = min(96, max(52, int(round(48 + severity_value * 9))))
        line_findings.append({
            "Source line": line_number if line_number is not None else "Not resolved",
            "Estimated issue likelihood": f"{likelihood}%",
            "Detected accessibility error": issue.get("title", issue.get("rule", "Accessibility issue")),
            "Severity": round(severity_value, 2),
            "HTML snippet": source_fragment or str(issue.get("element", ""))[:260],
            "Why flagged": issue.get("description", "Rule-based accessibility finding"),
        })

    if line_findings:
        line_df = pd.DataFrame(line_findings)
        line_df["_sort_line"] = pd.to_numeric(line_df["Source line"], errors="coerce").fillna(999999)
        line_df["_sort_likelihood"] = line_df["Estimated issue likelihood"].str.rstrip("%").astype(float)
        line_df = line_df.sort_values(["_sort_likelihood", "_sort_line"], ascending=[False, True])
        st.markdown(
            '<div class="risk-panel"><strong>⚠️ Highest-priority code locations</strong><br>'
            '<span>Higher percentages indicate stronger rule/severity evidence in this scan. They are heuristic priority estimates—not validated probabilities that a developer made a mistake.</span></div>',
            unsafe_allow_html=True,
        )
        top = line_df.iloc[0]
        top_cols = st.columns([1, 1, 2])
        top_cols[0].metric("Top estimated likelihood", top["Estimated issue likelihood"])
        top_cols[1].metric("Source line", str(top["Source line"]))
        top_cols[2].metric("Finding", str(top["Detected accessibility error"])[:60])
        display_df = line_df.drop(columns=["_sort_line", "_sort_likelihood"])
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=min(440, 100 + 54 * len(display_df)))
        with st.expander("How should I interpret these percentages?"):
            st.write(
                "The scanner first detects a specific accessibility rule failure. The percentage is a transparent severity-based priority indicator (based on the existing impact estimate), not a calibrated probability or a claim about the developer. "
                "Line numbers are best-effort matches against the HTML returned by the website; minified HTML, scripts that generate content, or repeated identical tags can make a line ambiguous."
            )
        st.caption("Note: a website URL usually exposes rendered HTML, not the developer's original project files. These are lines in the fetched HTML response where possible, not necessarily lines in the source repository.")
    else:
        st.success("No accessibility findings were available to rank. Run a scan with a page that returns readable HTML first.")

    st.markdown("#### Model signals behind the risk estimate")
    st.caption("These are the input patterns the Decision Tree used most often; they complement the line-level findings above.")
    importance_df = pd.DataFrame({
        "Input feature": RISK_FEATURE_NAMES,
        "Feature importance": risk_model.feature_importances_,
    }).sort_values("Feature importance", ascending=False)
    importance_df = importance_df[importance_df["Feature importance"] > 0]
    if not importance_df.empty:
        st.bar_chart(importance_df.set_index("Input feature"))
        st.dataframe(
            importance_df.assign(**{"Feature importance": importance_df["Feature importance"].round(4)}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("The model did not assign a non-zero importance to any displayed feature for this trained tree.")

    # ========================================================
    # PERFORMANCE AND EVALUATION METRICS
    # ========================================================
    st.divider()
    st.markdown("## 📊 Performance & Evaluation Metrics")
    st.markdown(
        "While traditional healthcare datasets evaluate via clinical metrics "
        "(such as Precision, Recall, Specificity, F1-Score, ROC-AUC, and Confusion Matrices), "
        "Accessify’s performance and evaluation metrics are quantified through information "
        "retrieval, classification confidence, and structural accessibility scoring."
    )

    evaluation_rows = [
        {
            "Evaluation area": "Information retrieval",
            "What it measures": "How closely a detected issue matches records in the accessibility dataset.",
            "Displayed indicator": "TF-IDF cosine similarity / closest matching dataset record, where available.",
            "Interpretation": "Higher similarity means the retrieved record is textually more relevant; it does not prove the match is correct."
        },
        {
            "Evaluation area": "Classification confidence",
            "What it measures": "The NLP model's confidence when flagging potentially unhelpful image alt text.",
            "Displayed indicator": "NLP classifier confidence, where a prediction is available.",
            "Interpretation": "This is a model score, not a validated probability of an accessibility violation."
        },
        {
            "Evaluation area": "Structural accessibility scoring",
            "What it measures": "A combined screening score based on detected accessibility issues and their estimated impact.",
            "Displayed indicator": "Accessify's 0–100 accessibility screening score.",
            "Interpretation": "Useful for prioritising fixes; it is not a WCAG certification or a formal compliance guarantee."
        },
        {
            "Evaluation area": "Risk and code-line prioritisation",
            "What it measures": "The relative priority of detected issues and the HTML lines associated with them.",
            "Displayed indicator": "Decision Tree risk class, severity, and estimated issue-likelihood priority indicator.",
            "Interpretation": "Line-level percentages are heuristic severity-based indicators, not calibrated statistical probabilities."
        },
    ]
    st.dataframe(pd.DataFrame(evaluation_rows), use_container_width=True, hide_index=True)
    st.caption(
        "These are application-level indicators, not a substitute for a labelled test set and a formal model evaluation. "
        "Precision, Recall, F1-Score, ROC-AUC, and a confusion matrix should only be reported after evaluating predictions "
        "against ground-truth labels that were not used to train the model. No such values are claimed here."
    )

