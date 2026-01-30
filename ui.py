import streamlit as st
from app import MultiAgentSystem

# --- Page Setup ---
st.set_page_config(page_title="AgentOS Executive", layout="wide", initial_sidebar_state="collapsed")

# --- Custom Styling ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    .stApp { background-color: #0e1117; color: #ffffff; }
    .main-title { font-size: 3rem; font-weight: 800; color: #00d4ff; margin-bottom: 0; }
    .log-text { font-family: monospace; color: #00ff41; font-size: 14px; background: #000; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AgentOS Executive</p>', unsafe_allow_html=True)
st.markdown("##### Autonomous Multi-Agent Research & Drafting Pipeline")
st.divider()

# Only the search input is visible
query = st.text_input("ENTER MISSION OBJECTIVE", placeholder="Search for a topic or define a goal...")

if st.button("🚀 EXECUTE SEARCH", use_container_width=True):
    if not query:
        st.warning("Please enter a search query.")
    else:
        system = MultiAgentSystem()
        state = {"query": query, "research": "", "critique": "", "email": "", "logs": []}
        
        # Results layout
        t1, t2, t3 = st.tabs(["🔍 INVESTIGATION", "⚖️ AUDIT", "📧 DELIVERABLE"])
        
        with st.status("📡 Orchestrating Agents...", expanded=True) as status:
            log_box = st.empty()
            
            # Agent Execution
            state = system.research_agent(state)
            log_box.markdown(f'<p class="log-text">{"<br>".join(state["logs"])}</p>', unsafe_allow_html=True)
            t1.markdown(state['research'])
            
            state = system.critic_agent(state)
            log_box.markdown(f'<p class="log-text">{"<br>".join(state["logs"])}</p>', unsafe_allow_html=True)
            t2.markdown(state['critique'])
            
            state = system.email_agent(state)
            log_box.markdown(f'<p class="log-text">{"<br>".join(state["logs"])}</p>', unsafe_allow_html=True)
            t3.markdown(state['email'])
            
            status.update(label="✅ TASK COMPLETED", state="complete", expanded=False)

        st.success("All agents have finished. Results are in the tabs above.")