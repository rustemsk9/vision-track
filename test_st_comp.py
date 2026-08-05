import streamlit as st
import streamlit.components.v1 as components
my_comp = components.declare_component("my_comp", path="frontend_2d")
my_comp()
