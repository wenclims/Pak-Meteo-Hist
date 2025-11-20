"""
Streamlit app
Functions related to building the page.
"""

import base64
import time

import extra_streamlit_components as stx
import folium
import streamlit as st
from pydantic.v1.utils import deep_update
from streamlit_folium import folium_static
from streamlit_js_eval import streamlit_js_eval

from app.forms import build_form, process_form
from app.plot import create_graph, display_context_info
from app.utils import get_form_defaults, get_query_params
from meteo_hist.base import MeteoHist


def build_menu() -> None:
    """
    Create the column holding the menu.
    """

    # Get query parameters
    query_params = get_query_params()

    if len(query_params) > 0:
        st.session_state["form_defaults"] = deep_update(
            st.session_state["form_defaults"], query_params
        )

    # Build form
    st.session_state["input_values"] = build_form(
        method="by_name", params=query_params
    )


def build_content(plot_placeholder, message_box) -> None:
    """
    Create the column holding the content.
    """

    # Save viewport width to session state
    st.session_state["viewport_width"] = streamlit_js_eval(
        js_expressions="window.innerWidth", key="ViewportWidth"
    )

    # Wait for viewport width to be set
    while (
        "viewport_width" not in st.session_state
        or st.session_state["viewport_width"] is None
    ):
        time.sleep(0.1)

    if st.session_state["input_values"] is not None:
        # Process form values
        input_processed = process_form(st.session_state["input_values"], message_box)

        # Make sure lat/lon values are set
        if isinstance(input_processed, dict) and not [
            x for x in (input_processed["lat"], input_processed["lon"]) if x is None
        ]:
            # Create figure for the graph
            plot_object, file_path = create_graph(input_processed, plot_placeholder)

            # Display some info about the data
            display_context_info(plot_object)

            # Display a download link
            try:
                with open(file_path, "rb") as file:
                    img_b64 = base64.b64encode(file.read()).decode()
                    st.markdown(
                        f'<a href="data:file/png;base64,{img_b64}" download="{file_path.split("/")[-1]}">Download file</a>',
                        unsafe_allow_html=True,
                    )
            except FileNotFoundError:
                st.write("File not found.")

            st.write("")

            # Show map
            with st.expander("Show map"):
                with st.spinner("Creating map..."):
                    folium_map = folium.Map(
                        location=[input_processed["lat"], input_processed["lon"]],
                        zoom_start=4,
                        height=500,
                    )
                    folium.Marker(
                        [input_processed["lat"], input_processed["lon"]],
                        popup=input_processed["location_name"],
                    ).add_to(folium_map)

                    # ✅ Added attribution to avoid ValueError
                    folium.TileLayer(
                        "Stamen Terrain",
                        attr="Map tiles by Stamen Design (CC BY 3.0), Data © OpenStreetMap contributors"
                    ).add_to(folium_map)

                    folium_static(folium_map)
