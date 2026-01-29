import streamlit as st
import requests

API_URL = "http://localhost:8000/run-agents"

st.set_page_config(
    page_title="Agent Orchestration Framework",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agent Orchestration Framework")
st.caption("Research • Critic • Summary • Email")

query = st.text_input(
    "Enter your research topic",
    placeholder="Example: Impact of AI in Healthcare"
)

if st.button("Run Agents 🚀") and query:

    with st.status("🔍 Agents are working...", expanded=True) as status:
        st.write("🔍 Research Agent running...")
        response = requests.post(API_URL, json={"query": query})
        data = response.json()

        st.write("🧠 Critic Agent reviewing...")
        st.write("📄 Summarizer Agent completed")
        st.write("📧 Email Agent completed")

        status.update(label="✅ All agents completed!", state="complete")

    st.divider()

    tabs = st.tabs([
        "🔍 Research",
        "🧠 Critic Review",
        "📄 Summary",
        "📊 Insights",
        "✅ Fact Check",
        "🏷️ Titles",
        "🔗 Sources",
        "📧 Email"
    ])

    critic_text = data["critic"]

    with tabs[0]:  # 🔍 Research
        st.markdown(data["research"])

    with tabs[1]:  # 🧠 Critic Review
        st.markdown(data["critic"])

    with tabs[2]:  # 📄 Summary
        st.markdown(data["summary"])

    with tabs[3]:  # 📊 Insights
        st.markdown(
            critic_text.split("Insights")[1].split("Titles")[0]
        )

    with tabs[4]:  # ✅ Fact Check
        st.markdown(
            critic_text.split("Fact Check")[1].split("Insights")[0]
        )

    with tabs[5]:  # 🏷️ Titles
        st.markdown(
            critic_text.split("Titles")[1].split("Sources")[0]
        )

    with tabs[6]:  # 🔗 Sources
        st.markdown(
            critic_text.split("Sources")[1]
        )

    with tabs[7]:  # 📧 Email
        st.text_area(
            "Generated Email",
            data["email"],
            height=260
        )