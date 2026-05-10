import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import zipfile

# ==============================
# PATHS (AUTO V1/V2 DETECTION)
# ==============================
BASE_PATH_V2 = Path("reports/shap_local_v2")
LLM_PATH_V2 = Path("reports/llm_local_v2/local_llm_explanations_v2.csv")

BASE_PATH_V1 = Path("reports/shap_local")
LLM_PATH_V1 = Path("reports/llm_local/local_llm_explanations.csv")

if LLM_PATH_V2.exists():
    BASE_PATH = BASE_PATH_V2
    LLM_PATH = LLM_PATH_V2
    VERSION = "v2"
else:
    BASE_PATH = BASE_PATH_V1
    LLM_PATH = LLM_PATH_V1
    VERSION = "v1"

# ==============================
# CLEAN TEXT
# ==============================
def clean_explanation(text):
    if isinstance(text, str):
        text = text.strip()
        if text.lower().startswith("certainly") and "1." in text:
            text = text[text.index("1."):]
        elif text.lower().startswith("certainly"):
            text = text.split("\n", 1)[-1]
    return text

# ==============================
# DISCRETIZATION
# ==============================
def discretize(endpoint, value):

    if pd.isna(value):
        return "Unknown"

    if endpoint in ["ADME_HLM", "ADME_RLM"]:
        if value <= 10:
            return "Low"
        elif value <= 40:
            return "Moderate"
        else:
            return "High"

    if endpoint in ["ADME_hPPB", "ADME_rPPB"]:
        if value <= 0.5:
            return "High binding"
        elif value <= 5:
            return "Intermediate"
        else:
            return "Low binding"

    if endpoint == "ADME_MDR1_ER":
        if value <= 2:
            return "Non-substrate"
        elif value <= 5:
            return "Moderate"
        else:
            return "Strong"

    if endpoint == "ADME_Sol":
        if value < 10:
            return "Poor"
        elif value < 100:
            return "Moderate"
        else:
            return "High"

    return "Unknown"

# ==============================
# PDF GENERATOR
# ==============================
def generate_pdf_bytes(endpoint, sample, explanation, mol_img_path, shap_img_path):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph(f"{endpoint} - {sample}", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    if mol_img_path.exists():
        elements.append(RLImage(str(mol_img_path), width=200, height=200))
        elements.append(Spacer(1, 10))

    if shap_img_path.exists():
        elements.append(RLImage(str(shap_img_path), width=300, height=200))
        elements.append(Spacer(1, 10))

    elements.append(Paragraph(explanation, styles["BodyText"]))

    doc.build(elements)

    buffer.seek(0)
    return buffer

# ==============================
# STREAMLIT
# ==============================
st.set_page_config(layout="wide")
st.title(f"ADME Local Explanations Explorer ({VERSION})")

# ==============================
# LOAD DATA
# ==============================
if not LLM_PATH.exists():
    st.error("CSV not found")
    st.stop()

df = pd.read_csv(LLM_PATH)

if df.empty:
    st.error("CSV vacío")
    st.stop()

st.success(f"Loaded {len(df)} explanations")

# ==============================
# SELECT ENDPOINT
# ==============================
endpoints = sorted(df["endpoint"].unique())
endpoint = st.selectbox("Endpoint", endpoints)

df_ep = df[df["endpoint"] == endpoint].copy()

# ==============================
# CATEGORY
# ==============================
df_ep["category"] = df_ep["prediction"].apply(lambda x: discretize(endpoint, x))

# ==============================
# FILTERS
# ==============================
colf1, colf2 = st.columns(2)

with colf1:
    sort_option = st.selectbox("Sort by prediction", ["None", "High → Low", "Low → High"])

with colf2:
    category_filter = st.selectbox("Filter category", ["All"] + sorted(df_ep["category"].unique()))

if sort_option == "High → Low":
    df_ep = df_ep.sort_values(by="prediction", ascending=False)
elif sort_option == "Low → High":
    df_ep = df_ep.sort_values(by="prediction", ascending=True)

if category_filter != "All":
    df_ep = df_ep[df_ep["category"] == category_filter]

# ==============================
# SEARCH
# ==============================
search = st.text_input("Search molecule")

samples = df_ep["sample"].tolist()

if search:
    samples = [s for s in samples if search.lower() in s.lower()]

if len(samples) == 0:
    st.warning("No molecules found")
    st.stop()

# ==============================
# SESSION STATE
# ==============================
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = samples[0]

if st.session_state.selected_sample not in samples:
    st.session_state.selected_sample = samples[0]

# ==============================
# SELECT SAMPLE
# ==============================
selected_sample = st.selectbox(
    "Molecule",
    samples,
    index=samples.index(st.session_state.selected_sample)
)

st.session_state.selected_sample = selected_sample

# ==============================
# NAVIGATION
# ==============================
idx = samples.index(selected_sample)

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    if idx > 0 and st.button("Previous"):
        st.session_state.selected_sample = samples[idx - 1]
        st.rerun()

with col_nav2:
    if idx < len(samples) - 1 and st.button("Next"):
        st.session_state.selected_sample = samples[idx + 1]
        st.rerun()

# ==============================
# GET ROW
# ==============================
row = df_ep[df_ep["sample"] == selected_sample].iloc[0]

# ==============================
# IMAGE PATHS
# ==============================
sample_path = BASE_PATH / endpoint / selected_sample
mol_img = sample_path / "molecule.png"
shap_img = sample_path / "waterfall.png"

# ==============================
# LAYOUT
# ==============================
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader("Molecule")
    if mol_img.exists():
        st.image(mol_img)
    else:
        st.warning("No molecule image")

with col2:
    st.subheader("SHAP")
    if shap_img.exists():
        st.image(shap_img)
    else:
        st.warning("No SHAP plot")

with col3:
    st.subheader("Explanation")
    explanation = clean_explanation(row["explanation"])
    st.write(explanation)

    pdf_bytes = generate_pdf_bytes(endpoint, selected_sample, explanation, mol_img, shap_img)

    st.download_button(
        "⬇️ Download PDF",
        data=pdf_bytes,
        file_name=f"{selected_sample}.pdf",
        mime="application/pdf"
    )

# ==============================
# EXPORT ZIP
# ==============================
st.markdown("---")
st.subheader("Export Top Molecules")

top_n = st.slider("Number of molecules", 5, 50, 20)

if st.button("Export TOP as ZIP"):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:

        df_top = df_ep.head(top_n)

        for _, r in df_top.iterrows():

            sample = r["sample"]
            explanation = clean_explanation(r["explanation"])

            sample_path = BASE_PATH / endpoint / sample
            mol_img = sample_path / "molecule.png"
            shap_img = sample_path / "waterfall.png"

            pdf_bytes = generate_pdf_bytes(endpoint, sample, explanation, mol_img, shap_img)
            z.writestr(f"{sample}.pdf", pdf_bytes.read())

    zip_buffer.seek(0)

    st.download_button(
        "Download ZIP",
        data=zip_buffer,
        file_name=f"{endpoint}_top_{top_n}.zip",
        mime="application/zip"
    )

# ==============================
# EXTRA INFO
# ==============================
st.markdown("---")
st.write("Prediction:", row.get("prediction"))
st.write("Category:", row.get("category"))
st.write("Features:", row.get("features"))
st.write("SHAP:", row.get("shap_values"))