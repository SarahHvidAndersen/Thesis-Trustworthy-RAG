import streamlit as st
from datetime import datetime

def render_chat_history(history):
    """
    Renders the chat history. Each turn should have keys: 'user', 'assistant', 'uncertainty', 'docs'.
    """
    for turn in history:
        # User message
        st.chat_message("user").write(turn.get("user", ""))

        # Assistant message with bubble wrapper
        uncertainty = turn.get("uncertainty")
        # Determine bubble class
        if uncertainty is not None:
            pct = int(round(uncertainty * 100))
            bubble_cls = (
                "conf-high" if pct >= 70 else
                "conf-med" if pct >= 40 else
                "conf-low"
            )
        else:
            bubble_cls = "conf-high"

        # Build HTML for the bubble
        html_parts = []
        # Response text
        response = turn.get("assistant", "")
        html_parts.append(f"<div>{response}</div>")

        # Confidence line
        if uncertainty is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            symbol = "✔️" if pct >= 70 else ("⚠️" if pct >= 40 else "❌")
            conf_html = (
                f"<div title='This score indicates how confident the chatbot is in its answer (0% = no confidence, 100% = fully confident).'"
                f" style='margin-top:4px;'>"
                f"<strong>{symbol} {pct}%</strong> "
                f"<small><i>{timestamp}</i></small>"
                f"</div>"
            )
            html_parts.append(conf_html)

        # Combine parts into bubble container
        bubble_html = (
            f"<div class='chat-bubble {bubble_cls}'>" +
            "".join(html_parts) +
            "</div>"
        )

        # Render as assistant message
        st.chat_message("assistant").markdown(bubble_html, unsafe_allow_html=True)

        # Retrieved documents
        docs = turn.get("docs", [])
        if docs:
            st.markdown("**Sources:**")
            for doc in docs:
                title = doc.get("metadata", {}).get("title") or doc.get("id")
                st.markdown(f"- {title}")

    st.markdown("---")
