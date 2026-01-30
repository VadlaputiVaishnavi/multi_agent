import streamlit as st
from app import MultiAgentSystem

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Research Assistant")

@st.cache_resource
def load_system():
    return MultiAgentSystem()

try:
    system = load_system()
except Exception as e:
    st.error(str(e))
    st.stop()

query = st.text_input(
    "Enter your research topic:",
    placeholder="Impact of AI on software engineering jobs",
)

if st.button("🚀 Run Agents"):
    if not query:
        st.warning("Please enter a topic.")
    else:
        state = {
            "query": query,
            "research": "",
            "critique": "",
            "email": "",
            "logs": [],
        }

        with st.spinner("🔍 Researching..."):
            state = system.research_agent(state)

        with st.spinner("⚖️ Reviewing..."):
            state = system.critic_agent(state)

        with st.spinner("📧 Writing email..."):
            state = system.email_agent(state)

        st.success("Done!")

        tab1, tab2, tab3 = st.tabs(["📚 Research", "⚖️ Critique", "📧 Email"])

        with tab1:
            st.markdown(state["research"])

        with tab2:
            st.markdown(state["critique"])

        with tab3:
            st.markdown(state["email"])

        with st.expander("Execution Logs"):
            for log in state["logs"]:
                st.write(log)
