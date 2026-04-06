import streamlit as st
import pandas as pd
from env.data_env import DataCleaningEnv
from env.models import Action
from agent import DataCleaningAgent
from env.grader import Grader

# =================== Page Config ===================
st.set_page_config(page_title="AI Data Cleaning Environment", layout="wide")

# =================== Styling ===================
st.markdown("""
<style>
.stApp {
    background: #FFF8F0;
    color: #1F2937;
}
.title {
    text-align: center;
    color: #E65100;
    font-size: 44px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #374151;
    font-size: 18px;
}
.section1 { color: #E65100; }
.section2 { color: #6A1B9A; }
.section3 { color: #1565C0; }
.section4 { color: #2E7D32; }
.section5 { color: #C62828; }
.section6 { color: #00897B; }

.section {
    font-size: 26px;
    font-weight: bold;
    margin-top: 25px;
}
.card {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.timeline {
    border-left: 5px solid #FF6600;
    padding-left: 15px;
    margin: 10px 0;
}
div.stButton > button {
    background: linear-gradient(90deg, #FF7A18, #FFB347);
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =================== Header ===================
st.markdown('<div class="title">AI Data Cleaning Environment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart AI + Manual Data Cleaning Dashboard</div>', unsafe_allow_html=True)

# =================== Sidebar ===================
task = st.sidebar.selectbox("Dataset Difficulty", ["easy", "medium", "hard"])

# =================== Initialize ===================
if "env" not in st.session_state or st.session_state.get("task") != task:
    st.session_state.task = task
    st.session_state.env = DataCleaningEnv(task)
    st.session_state.obs = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.agent = DataCleaningAgent()
    st.session_state.timeline = []

env = st.session_state.env
obs = st.session_state.obs
agent = st.session_state.agent

# =================== Dataset Overview ===================
st.markdown('<div class="section section1">Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", obs.rows)
c2.metric("Columns", len(obs.columns))
c3.metric("Missing", obs.missing_values)
c4.metric("Duplicates", obs.duplicates)

# =================== Dataset ===================
st.markdown('<div class="section section2">Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(env.df, use_container_width=True)

# =================== Progress ===================
st.markdown('<div class="section section6">Cleaning Progress</div>', unsafe_allow_html=True)

total_cells = env.df.size
missing_cells = env.df.isnull().sum().sum()
progress = 1 - (missing_cells / total_cells)

st.progress(progress)
st.write(f"Progress: {round(progress * 100, 2)}%")

# =================== Manual ===================
st.markdown('<div class="section section3">Manual Cleaning</div>', unsafe_allow_html=True)

action_type = st.selectbox("Action", [
    "remove_duplicates",
    "fill_missing_mean",
    "fill_missing_median",
    "fill_missing_mode",
    "convert_to_numeric",
    "normalize_column",
])

column = None
if action_type != "remove_duplicates":
    column = st.selectbox("Column", obs.columns)

if st.button("Run Action"):
    obs, reward, done, _ = env.step(Action(action_type=action_type, column=column))
    st.session_state.obs = obs
    st.session_state.done = done
    st.session_state.timeline.append((action_type, column, reward.value))
    st.success(f"Reward: {reward.value}")
    st.rerun()

# =================== Agent ===================
st.markdown('<div class="section section4">AI Agent Cleaning</div>', unsafe_allow_html=True)

if st.button("Run Agent"):
    obs, rewards = agent.clean_dataset(env)
    st.session_state.obs = obs
    st.session_state.done = True

    for action_type, column, value, _ in rewards:
        st.session_state.timeline.append((action_type, column, value))

    st.success("Agent completed cleaning!")

# ===================  CLEANED DATASET (NEW) ===================
if st.session_state.done:
    st.markdown('<div class="section section2">Cleaned Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(env.df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =================== Timeline ===================
st.markdown('<div class="section section5">Action Timeline</div>', unsafe_allow_html=True)

for i, (act, col, val) in enumerate(st.session_state.timeline, 1):
    st.markdown(f"""
    <div class="timeline">
        <b>Step {i}</b><br>
        Action: {act}<br>
        Column: {col}<br>
        Reward: {val}
    </div>
    """, unsafe_allow_html=True)

# =================== Grader ===================
st.markdown('<div class="section section6">Grader Score</div>', unsafe_allow_html=True)
score = Grader(env.df).evaluate()

st.markdown(f"""
<div class="card">
<h2 style="color:#E65100;">Final Score: {score*100:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

# =================== Download ===================
st.download_button(
    "Download Cleaned Dataset",
    env.df.to_csv(index=False),
    file_name="cleaned_data.csv"
)

# =================== Metrics ===================
st.markdown('<div class="section section1">Cleaning Metrics</div>', unsafe_allow_html=True)

metrics = []
for col in env.df.columns:
    total = len(env.df)
    miss = env.df[col].isnull().sum()
    dup = env.df.duplicated(subset=[col]).sum()
    miss_pct = miss / total * 100
    score_val = max(0, 1 - miss_pct/100 - dup/total)

    metrics.append({
        "Column": col,
        "Missing %": round(miss_pct,1),
        "Duplicates": dup,
        "Score": round(score_val,2)
    })

df_metrics = pd.DataFrame(metrics)

def color_score(v):
    if v >= 0.8:
        return "background-color:#C8E6C9;"
    elif v >= 0.5:
        return "background-color:#FFE0B2;"
    else:
        return "background-color:#FFCDD2;"

styled = df_metrics.style.map(color_score, subset=["Score"])

st.dataframe(styled, use_container_width=True)