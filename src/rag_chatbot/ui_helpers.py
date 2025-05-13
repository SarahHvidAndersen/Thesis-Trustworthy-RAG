# the smaller font for the sources just doesn't work - maybe fix later

import streamlit as st
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from html import escape


def _inject_css() -> None:
    """Insert once‑per‑session CSS for chat bubbles and source dropdowns."""
    if st.session_state.get("__ui_helpers_css_injected"):
        return

    st.session_state["__ui_helpers_css_injected"] = True

    st.markdown(
        """
        <style>
            .chat-bubble {
                position: relative;
                padding: 0.75rem 1rem;
                border-radius: 0.75rem;
                margin-bottom: 0.5rem;
                background: var(--secondary-background);
            }
            .conf-high { border-left: 6px solid #2ecc71; }
            .conf-med  { border-left: 6px solid #f1c40f; }
            .conf-low  { border-left: 6px solid #e74c3c; }

            /* Footer */
            .bubble-footer {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-top: 0.5rem;
                gap: 0.5rem;
                font-size: 0.8rem; /* smaller than main text */
            }
            .conf-info { flex: 1 1 auto; }

            /* ↓↓↓ Force smaller font‑size for ALL content in Sources ↓↓↓ */
            .chat-bubble details.source-root,
            .chat-bubble details.source-root *,
            .chat-bubble details.source-item,
            .chat-bubble details.source-item * {
                font-size: 0.8rem !important; /* override Streamlit defaults */
                line-height: 1.2;
            }

            /* Minor spacing tweaks */
            details.source-item { margin-top: 0.25rem; }
            details.source-item > p { margin: 0.25rem 0 0.25rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history(history: List[Dict]) -> None:
    """Render chat bubbles with confidence star and compact Sources dropdown."""

    _inject_css()

    for turn in history:
        # User message
        st.chat_message("user").write(turn.get("user", ""))

        # Confidence calculation
        conf = turn.get("calibrated_confidence")
        pct: int | None = int(round(conf * 100)) if conf is not None else None
        bubble_cls = (
            "conf-high" if (pct or 0) >= 70 else "conf-med" if (pct or 0) >= 30 else "conf-low"
        )

        # Assistant body
        body_html = f"<div class='bubble-body'>{turn.get('assistant', '')}</div>"

        # Confidence info
        conf_html = ""
        if pct is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            conf_html = (
                f"<div class='conf-info' title='Model confidence'>"
                f"<strong>⍟ {pct}%</strong> <small><i>{ts}</i></small></div>"
            )

        # Sources dropdown
        docs = turn.get("docs", [])
        sources_html = ""
        if docs:
            # Group by (title, author)
            grouped: Dict[tuple[str, str], List[str]] = defaultdict(list)
            for doc in docs:
                meta   = doc.get("metadata", {})
                title  = meta.get("title") or doc.get("id", "Untitled source")
                author = meta.get("author", "")
                grouped[(title, author)].append(doc.get("text", "").strip())

            # Build inner collapsibles
            inner_parts: List[str] = []
            for (title, author), snippets in grouped.items():
                header_safe = escape(title)
                if author:
                    header_safe += " — " + escape(author)

                # Build snippet paragraphs (escaped)
                snippet_paras = "".join(
                    f"<p><b>Snippet {i}:</b> {escape(s.replace('\n', ' '))}</p>" for i, s in enumerate(snippets, 1)
                )

                inner_parts.append(
                    f"<details class='source-item'><summary>{header_safe}</summary>{snippet_paras}</details>"
                )

            inner_html = "".join(inner_parts)
            sources_html = (
                f"<details class='source-root'><summary>Sources ({len(docs)})</summary>{inner_html}</details>"
            )

        # Footer (flex container)
        footer_html = (
            f"<div class='bubble-footer'>{conf_html}{sources_html}</div>" if (conf_html or sources_html) else ""
        )

        # Final bubble
        bubble_html = (
            f"<div class='chat-bubble {bubble_cls}'>{body_html}{footer_html}</div>"
        )

        st.chat_message("assistant").markdown(bubble_html, unsafe_allow_html=True)

    # Separator
    st.markdown("---")