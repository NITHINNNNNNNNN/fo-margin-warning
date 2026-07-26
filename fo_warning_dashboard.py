"""
F&O Margin Early Warning System — Live Dashboard
Run: python fo_warning_dashboard.py
Requirements: pip install dash plotly yfinance pandas
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── WARNING MODEL THRESHOLDS ─────────────────────────────────────────────────
VIX_THRESHOLD   = 20.0   # VIX above this = fear zone
HVOL_THRESHOLD  = 18.0   # 10D annualised vol above this = turbulent
VIX5D_THRESHOLD = 5.0    # VIX 5-day change above this = fear spike

# ── LOAD HISTORICAL MODEL DATA ───────────────────────────────────────────────
def load_history():
    try:
        df = pd.read_csv('NSE_FO_Final_Model.csv', dtype=str)
        num = ['VIX_Level','VIX_5D_Change_Pct','Nifty_10D_HVol',
               'Nifty_10D_Return','BankNifty_5D_HVol','Warning_Score']
        for c in num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return pd.DataFrame()

# ── FETCH LIVE SIGNALS ───────────────────────────────────────────────────────
def fetch_live_signals():
    end   = datetime.today()
    start = end - timedelta(days=30)
    try:
        nifty  = yf.download('^NSEI',     start=start, end=end, progress=False, auto_adjust=True)
        vix    = yf.download('^INDIAVIX', start=start, end=end, progress=False, auto_adjust=True)
        bnifty = yf.download('^NSEBANK',  start=start, end=end, progress=False, auto_adjust=True)

        def flatten(d):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            return d

        nifty  = flatten(nifty)
        vix    = flatten(vix)
        bnifty = flatten(bnifty)

        vix_level    = float(vix['Close'].iloc[-1])
        vix_5d       = float((vix['Close'].iloc[-1] - vix['Close'].iloc[-6]) / vix['Close'].iloc[-6] * 100)
        nifty_ret    = nifty['Close'].pct_change().dropna().tail(10)
        nifty_hvol   = float(nifty_ret.std() * (252**0.5) * 100)
        nifty_10d    = float((nifty['Close'].iloc[-1] - nifty['Close'].iloc[-11]) / nifty['Close'].iloc[-11] * 100)
        bn_ret       = bnifty['Close'].pct_change().dropna().tail(5)
        bn_hvol      = float(bn_ret.std() * (252**0.5) * 100)
        nifty_price  = float(nifty['Close'].iloc[-1])
        as_of        = str(nifty.index[-1].date())

        return {
            'vix_level':   round(vix_level, 2),
            'vix_5d':      round(vix_5d, 2),
            'nifty_hvol':  round(nifty_hvol, 2),
            'nifty_10d':   round(nifty_10d, 2),
            'bn_hvol':     round(bn_hvol, 2),
            'nifty_price': round(nifty_price, 2),
            'as_of':       as_of,
            'error':       None
        }
    except Exception as e:
        return {'error': str(e)}

def compute_signal(signals):
    s1 = 1 if signals['vix_level']  >= VIX_THRESHOLD   else 0
    s2 = 1 if signals['nifty_hvol'] >= HVOL_THRESHOLD  else 0
    s3 = 1 if signals['vix_5d']     >= VIX5D_THRESHOLD else 0
    score = s1 + s2 + s3
    if score >= 2:
        return 'RED', '🔴 HIGH RISK — Margin Hike Likely', '#ff4444', score
    elif score == 1:
        return 'AMBER', '🟡 WATCH — Elevated Conditions', '#ffaa00', score
    else:
        return 'GREEN', '🟢 LOW RISK — Conditions Normal', '#00cc66', score

# ── APP LAYOUT ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='F&O Margin Early Warning System')


server = app.server

app.layout = html.Div([

    # Header
    html.Div([
        html.H1('F&O Margin Early Warning System',
                style={'color':'white','margin':'0','fontSize':'24px'}),
        html.P('Predicts NSE/SEBI F&O margin revision risk using VIX, volatility & fear signals',
               style={'color':'#aaa','margin':'4px 0 0 0','fontSize':'13px'}),
    ], style={'background':'#1a1a2e','padding':'20px 30px','borderBottom':'2px solid #16213e'}),

    # Refresh button + last updated
    html.Div([
        html.Button('🔄 Refresh Live Data', id='refresh-btn',
                    style={'background':'#0f3460','color':'white','border':'none',
                           'padding':'10px 20px','borderRadius':'6px','cursor':'pointer','fontSize':'14px'}),
        html.Span(id='last-updated', style={'color':'#888','marginLeft':'20px','fontSize':'13px'}),
    ], style={'padding':'15px 30px','background':'#16213e'}),

    # Warning Banner
    html.Div(id='warning-banner', style={'padding':'0 30px'}),

    # KPI Cards Row
    html.Div(id='kpi-cards', style={
        'display':'flex','gap':'15px','padding':'20px 30px','flexWrap':'wrap'
    }),

    # Charts Row
    html.Div([
        html.Div([
            html.H3('Signal Gauge', style={'color':'white','fontSize':'16px','marginBottom':'10px'}),
            dcc.Graph(id='gauge-chart', config={'displayModeBar':False})
        ], style={'flex':'1','background':'#16213e','padding':'20px','borderRadius':'8px'}),

        html.Div([
            html.H3('Historical Hike Events vs VIX', style={'color':'white','fontSize':'16px','marginBottom':'10px'}),
            dcc.Graph(id='history-chart', config={'displayModeBar':False})
        ], style={'flex':'2','background':'#16213e','padding':'20px','borderRadius':'8px'}),
    ], style={'display':'flex','gap':'15px','padding':'0 30px 20px'}),

    # Signal Conditions Table
    html.Div([
        html.H3('Signal Conditions Detail', style={'color':'white','fontSize':'16px','marginBottom':'10px'}),
        html.Div(id='signal-table')
    ], style={'margin':'0 30px 20px','background':'#16213e','padding':'20px','borderRadius':'8px'}),

    # Footer
    html.Div([
        html.P('Built by Nithin M | B.Com FinTech, VIT Vellore | Based on 163 NSE F&O margin circulars (2018-2026)',
               style={'color':'#555','fontSize':'12px','margin':'0','textAlign':'center'})
    ], style={'padding':'15px','borderTop':'1px solid #222'}),

    dcc.Store(id='signals-store'),
    dcc.Interval(id='auto-refresh', interval=300*1000, n_intervals=0),

], style={'background':'#0f0f1a','minHeight':'100vh','fontFamily':'Inter, sans-serif'})


# ── CALLBACKS ────────────────────────────────────────────────────────────────
@app.callback(
    Output('signals-store','data'),
    Output('last-updated','children'),
    Input('refresh-btn','n_clicks'),
    Input('auto-refresh','n_intervals'),
    prevent_initial_call=False
)
def update_signals(n, interval):
    signals = fetch_live_signals()
    ts = f"Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}"
    return signals, ts


@app.callback(
    Output('warning-banner','children'),
    Output('kpi-cards','children'),
    Output('gauge-chart','figure'),
    Output('history-chart','figure'),
    Output('signal-table','children'),
    Input('signals-store','data'),
)
def update_ui(signals):
    if not signals or signals.get('error'):
        err = signals.get('error','Unknown error') if signals else 'Loading...'
        banner = html.Div(f'⚠️ Could not fetch live data: {err}',
                          style={'background':'#333','color':'#fff','padding':'15px','borderRadius':'6px','margin':'10px 0'})
        return banner, [], go.Figure(), go.Figure(), []

    color_code, label, hex_color, score = compute_signal(signals)

    # Warning Banner
    banner = html.Div([
        html.H2(label, style={'margin':'0','fontSize':'20px','color':'white'}),
        html.P(f"Warning Score: {score}/3 signals active | As of {signals.get('as_of','')}",
               style={'margin':'5px 0 0 0','color':'rgba(255,255,255,0.8)','fontSize':'13px'})
    ], style={'background':hex_color,'padding':'15px 25px','borderRadius':'8px',
              'margin':'15px 0 0 0','boxShadow':f'0 0 20px {hex_color}88'})

    # KPI Cards
    kpis = [
        ('VIX Level', signals['vix_level'], f"Threshold: {VIX_THRESHOLD}",
         '#ff4444' if signals['vix_level'] >= VIX_THRESHOLD else '#00cc66'),
        ('VIX 5D Change', f"{signals['vix_5d']:+.2f}%", f"Threshold: +{VIX5D_THRESHOLD}%",
         '#ff4444' if signals['vix_5d'] >= VIX5D_THRESHOLD else '#00cc66'),
        ('Nifty 10D HVol', f"{signals['nifty_hvol']}%", f"Threshold: {HVOL_THRESHOLD}%",
         '#ff4444' if signals['nifty_hvol'] >= HVOL_THRESHOLD else '#00cc66'),
        ('Nifty 10D Return', f"{signals['nifty_10d']:+.2f}%", 'Market direction', '#4e9af1'),
        ('BankNifty 5D Vol', f"{signals['bn_hvol']}%", 'Banking sector stress', '#9b59b6'),
        ('Nifty50 Price', f"₹{signals['nifty_price']:,.0f}", 'Current level', '#f39c12'),
    ]

    cards = []
    for title, value, sub, color in kpis:
        cards.append(html.Div([
            html.P(title, style={'color':'#888','margin':'0','fontSize':'12px'}),
            html.H3(str(value), style={'color':color,'margin':'5px 0','fontSize':'22px'}),
            html.P(sub, style={'color':'#555','margin':'0','fontSize':'11px'}),
        ], style={'background':'#1a1a2e','padding':'15px 20px','borderRadius':'8px',
                  'borderLeft':f'3px solid {color}','minWidth':'140px','flex':'1'}))

    # Gauge Chart
    gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        title={'text':'Warning Score','font':{'color':'white','size':14}},
        number={'font':{'color':hex_color,'size':40}},
        gauge={
            'axis':{'range':[0,3],'tickcolor':'white','tickfont':{'color':'white'}},
            'bar':{'color':hex_color,'thickness':0.3},
            'bgcolor':'#1a1a2e',
            'bordercolor':'#333',
            'steps':[
                {'range':[0,1],'color':'#1a3a1a'},
                {'range':[1,2],'color':'#3a3a1a'},
                {'range':[2,3],'color':'#3a1a1a'},
            ],
            'threshold':{'line':{'color':hex_color,'width':4},'value':score}
        }
    ))
    gauge.update_layout(
        paper_bgcolor='#16213e', plot_bgcolor='#16213e',
        font={'color':'white'}, height=250, margin=dict(t=40,b=20,l=20,r=20)
    )

    # History Chart
    hist_df = load_history()
    fig = go.Figure()

    if not hist_df.empty and 'Date' in hist_df.columns:
        def get_dir(s):
            s = str(s).upper()
            if any(w in s for w in ['ADDITIONAL','INCREASE','ENHANCE','HIKE','IMPOSE','RISK MANAGEMENT MEASURES']):
                return 'Hike'
            return 'Other'

        hist_df['Dir'] = hist_df['Subject'].apply(get_dir)
        hikes = hist_df[hist_df['Dir'] == 'Hike']
        others = hist_df[hist_df['Dir'] != 'Hike']

        fig.add_trace(go.Scatter(
            x=others['Date'], y=others['VIX_Level'],
            mode='markers', name='No Hike',
            marker=dict(color='#4e9af1', size=6, opacity=0.5)
        ))
        fig.add_trace(go.Scatter(
            x=hikes['Date'], y=hikes['VIX_Level'],
            mode='markers', name='Hike Event',
            marker=dict(color='#ff4444', size=9, symbol='triangle-up')
        ))
        fig.add_hline(y=VIX_THRESHOLD, line_dash='dash',
                      line_color='orange', annotation_text=f'VIX Threshold ({VIX_THRESHOLD})')

    fig.update_layout(
        paper_bgcolor='#16213e', plot_bgcolor='#1a1a2e',
        font={'color':'white'}, height=250,
        xaxis=dict(color='white', gridcolor='#333'),
        yaxis=dict(color='white', gridcolor='#333', title='VIX Level'),
        legend=dict(bgcolor='#1a1a2e', bordercolor='#333'),
        margin=dict(t=20,b=40,l=50,r=20)
    )

    # Signal Table
    rows = [
        ('Signal 1', f'VIX Level ≥ {VIX_THRESHOLD}', signals['vix_level'],
         '✅ ACTIVE' if signals['vix_level'] >= VIX_THRESHOLD else '⬜ inactive'),
        ('Signal 2', f'Nifty 10D HVol ≥ {HVOL_THRESHOLD}%', f"{signals['nifty_hvol']}%",
         '✅ ACTIVE' if signals['nifty_hvol'] >= HVOL_THRESHOLD else '⬜ inactive'),
        ('Signal 3', f'VIX 5D Change ≥ +{VIX5D_THRESHOLD}%', f"{signals['vix_5d']:+.2f}%",
         '✅ ACTIVE' if signals['vix_5d'] >= VIX5D_THRESHOLD else '⬜ inactive'),
    ]

    table = html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={'color':'#888','padding':'8px 12px','textAlign':'left','fontSize':'12px'})
            for h in ['Signal','Condition','Current Value','Status']
        ])),
        html.Tbody([
            html.Tr([
                html.Td(r[0], style={'color':'white','padding':'10px 12px','fontSize':'13px'}),
                html.Td(r[1], style={'color':'#aaa','padding':'10px 12px','fontSize':'13px'}),
                html.Td(str(r[2]), style={'color':'#4e9af1','padding':'10px 12px','fontSize':'13px','fontWeight':'bold'}),
                html.Td(r[3], style={
                    'padding':'10px 12px','fontSize':'13px',
                    'color':'#ff4444' if 'ACTIVE' in r[3] else '#555'
                }),
            ], style={'borderBottom':'1px solid #222'})
            for r in rows
        ])
    ], style={'width':'100%','borderCollapse':'collapse'})

    return banner, cards, gauge, fig, table


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  F&O MARGIN EARLY WARNING SYSTEM")
    print("="*55)
    print("  Dashboard starting...")
    print("  Open browser: http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
