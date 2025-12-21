import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the Brain
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define the Logic
def calculate_resonance(text1, text2):
    # SCRUBBER: Remove invisible spaces/newlines
    text1 = text1.strip()
    text2 = text2.strip()
    
    # Encode
    embeddings = model.encode([text1, text2])
    
    # Calculate Score
    score = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    
    # The Director's Verdict
    if score > 0.85:
        verdict = "⚠️ PARROT (Too Similar)"
    elif score > 0.65:
        verdict = "🔥 HIGH FIDELITY (Strong Lock)"
    elif score > 0.40:
        verdict = "✅ STABLE RESONANCE (The Sweet Spot)"
    elif score > 0.20:
        verdict = "🌊 DRIFTING (Weak Signal)"
    else:
        verdict = "❌ COLLAPSE (No Connection)"
        
    return f"{score:.3f}", verdict

# 3. Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ The Resonator")
    gr.Markdown("### FlameTeam Semantic Alignment Detector")
    
    with gr.Row():
        box1 = gr.Textbox(label="Turn A (Your Input)", lines=4, placeholder="Paste your prompt here...")
        box2 = gr.Textbox(label="Turn B (AI Response)", lines=4, placeholder="Paste the response here...")
    
    btn = gr.Button("Calculate Resonance", variant="primary")
    
    with gr.Row():
        out_score = gr.Label(label="Similarity Score")
        out_verdict = gr.Label(label="System Status")

    btn.click(calculate_resonance, inputs=[box1, box2], outputs=[out_score, out_verdict])

# 4. Launch
app.launch()
