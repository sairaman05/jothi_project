import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram(output_path="results/system_architecture_diagram.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    bg_main = '#f8fafc'
    fig.patch.set_facecolor(bg_main)
    ax.set_facecolor(bg_main)

    col_stream = '#3b82f6'    # Blue
    col_cache = '#10b981'     # Emerald Green
    col_agent1 = '#8b5cf6'    # Purple
    col_agent2 = '#f59e0b'    # Amber
    col_agent3 = '#06b6d4'    # Cyan
    col_mem = '#ec4899'       # Pink/Rose
    col_border = '#334155'    # Slate 700

    # Title Banner
    ax.text(7.5, 9.6, "KGARevion + MedStreamMem: End-to-End System Architecture", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#0f172a')
    ax.text(7.5, 9.2, "Neuro-Symbolic Reasoning Pipeline with Domain-Weighted Bounded Streaming Memory", 
            ha='center', va='center', fontsize=12, fontstyle='italic', color='#475569')

    # 1. Input Box (Clinical Query Stream)
    rect_in = patches.FancyBboxPatch((0.5, 6.2), 3.0, 2.2, boxstyle="round,pad=0.2", 
                                    fc='#eff6ff', ec=col_stream, lw=2.0)
    ax.add_patch(rect_in)
    ax.text(2.0, 7.9, "1. Clinical Query Stream", ha='center', va='center', fontsize=12, fontweight='bold', color=col_stream)
    ax.text(2.0, 7.3, "• Clinical Inquiry (MCQ/SAQ)\n• Clinical Authority Weight:\n  tau in [0.48, 0.98]\n• High/Mid/Low Trust Tiers", 
            ha='center', va='center', fontsize=9.5, color='#1e293b')

    # Arrow 1: Stream to Cache Check
    ax.annotate('', xy=(4.2, 7.3), xytext=(3.7, 7.3),
                arrowprops=dict(facecolor=col_border, edgecolor=col_border, width=2, headwidth=8))

    # 2. MedStreamMem Cache Interception
    rect_cache = patches.FancyBboxPatch((4.4, 5.8), 3.4, 3.0, boxstyle="round,pad=0.2", 
                                       fc='#ecfdf5', ec=col_cache, lw=2.2)
    ax.add_patch(rect_cache)
    ax.text(6.1, 8.4, "2. MedStreamMem Cache", ha='center', va='center', fontsize=12, fontweight='bold', color='#047857')
    ax.text(6.1, 7.6, "O(1) Hash Table Lookup\nQuery Normalization & Match", ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46')
    
    # Hit / Miss Branches
    # Hit Branch (Up and Right to Instant Return)
    ax.annotate('', xy=(8.5, 8.0), xytext=(8.0, 8.0),
                arrowprops=dict(facecolor='#059669', edgecolor='#059669', width=2, headwidth=8))
    rect_hit = patches.FancyBboxPatch((8.7, 7.1), 5.8, 1.8, boxstyle="round,pad=0.2", 
                                     fc='#d1fae5', ec='#059669', lw=2.0)
    ax.add_patch(rect_hit)
    ax.text(11.6, 8.3, "[CACHE HIT] Instant Retrieval (< 0.0001s)", ha='center', va='center', fontsize=11, fontweight='bold', color='#065f46')
    ax.text(11.6, 7.6, "• Bypass LLM Inference entirely (50s -> 0.0001s)\n• Increment Access Frequency: phi += 1\n• Reset Elapsed Inactivity: delta_t = 0\n• Return Cached Clinical Answer & Verified Triplets", 
            ha='center', va='center', fontsize=9, color='#064e3b')

    # Miss Branch (Down to KGARevion Core)
    ax.text(6.1, 5.5, "[CACHE MISS]", ha='center', va='center', fontsize=10, fontweight='bold', color='#b91c1c')
    ax.annotate('', xy=(6.1, 4.8), xytext=(6.1, 5.6),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=2, headwidth=8))

    # 3. Large Container for KGARevion Neuro-Symbolic Engine
    rect_kga = patches.FancyBboxPatch((0.5, 0.6), 8.8, 3.9, boxstyle="round,pad=0.3", 
                                     fc='#ffffff', ec='#94a3b8', lw=1.8, linestyle='--')
    ax.add_patch(rect_kga)
    ax.text(4.9, 4.25, "3. KGARevion Neuro-Symbolic Multi-Agent Pipeline", ha='center', va='center', fontsize=12, fontweight='bold', color='#334155')

    # Agent 1: Triplet Generator
    rect_a1 = patches.FancyBboxPatch((0.8, 1.1), 2.5, 2.7, boxstyle="round,pad=0.15", 
                                    fc='#f5f3ff', ec=col_agent1, lw=1.8)
    ax.add_patch(rect_a1)
    ax.text(2.05, 3.4, "Agent 1: Generator", ha='center', va='center', fontsize=11, fontweight='bold', color=col_agent1)
    ax.text(2.05, 2.3, "Dynamic Triplet\nExtraction\n(action/generate.py)\n\n• Entity Detection\n• Candidate (h, r, t)\n  Relationships\n• Clinical Context", 
            ha='center', va='center', fontsize=8.5, color='#1e293b')

    # Arrow A1 -> A2
    ax.annotate('', xy=(3.9, 2.45), xytext=(3.5, 2.45),
                arrowprops=dict(facecolor=col_border, edgecolor=col_border, width=2, headwidth=7))

    # Agent 2: Triplet Reviewer / Classifier
    rect_a2 = patches.FancyBboxPatch((4.1, 1.1), 2.5, 2.7, boxstyle="round,pad=0.15", 
                                    fc='#fffbeb', ec=col_agent2, lw=1.8)
    ax.add_patch(rect_a2)
    ax.text(5.35, 3.4, "Agent 2: Reviewer", ha='center', va='center', fontsize=11, fontweight='bold', color='#b45309')
    ax.text(5.35, 2.3, "Verification &\nGraph Pruning\n(action/review.py)\n\n• PrimeKG Alignment\n• Structural Scoring\n• Prune Hallucinations\n  (Score >= 0.50)", 
            ha='center', va='center', fontsize=8.5, color='#1e293b')

    # Arrow A2 -> A3
    ax.annotate('', xy=(7.2, 2.45), xytext=(6.8, 2.45),
                arrowprops=dict(facecolor=col_border, edgecolor=col_border, width=2, headwidth=7))

    # Agent 3: Answer Synthesizer
    rect_a3 = patches.FancyBboxPatch((7.4, 1.1), 2.5, 2.7, boxstyle="round,pad=0.15", 
                                    fc='#ecfeff', ec=col_agent3, lw=1.8)
    ax.add_patch(rect_a3)
    ax.text(8.65, 3.4, "Agent 3: Synthesizer", ha='center', va='center', fontsize=11, fontweight='bold', color='#0891b2')
    ax.text(8.65, 2.3, "Grounded Medical\nAnswer Synthesis\n(action/answer.py)\n\n• Conditioned on\n  Verified Triplets\n• Differential Diagnosis\n• Choice (A/B/C/D)", 
            ha='center', va='center', fontsize=8.5, color='#1e293b')

    # Arrow from KGARevion to MedStreamMem Memory Buffer
    ax.annotate('', xy=(10.6, 2.45), xytext=(10.1, 2.45),
                arrowprops=dict(facecolor=col_border, edgecolor=col_border, width=2, headwidth=8))
    ax.text(10.35, 2.75, "Answer +\nTriplets", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e293b')

    # 4. MedStreamMem Bounded Eviction Engine Box
    rect_mem = patches.FancyBboxPatch((10.7, 0.6), 3.8, 5.8, boxstyle="round,pad=0.25", 
                                     fc='#fdf2f8', ec=col_mem, lw=2.2)
    ax.add_patch(rect_mem)
    ax.text(12.6, 6.0, "4. MedStreamMem Eviction Engine", ha='center', va='center', fontsize=12, fontweight='bold', color='#be185d')
    ax.text(12.6, 5.4, "Capacity: K = 50 Items | Bound: O(K)", ha='center', va='center', fontsize=10, fontweight='bold', color='#831843')

    # Formula Box inside Memory Engine
    rect_form = patches.FancyBboxPatch((11.0, 4.0), 3.2, 1.1, boxstyle="round,pad=0.1", 
                                      fc='#ffffff', ec='#f472b6', lw=1.5)
    ax.add_patch(rect_form)
    ax.text(12.6, 4.65, "Domain-Weighted Priority Score:", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#9d174d')
    ax.text(12.6, 4.25, r"$S_i = \frac{\phi_i \cdot \tau_i}{\Delta t_i + 1.0}$", ha='center', va='center', fontsize=12, fontweight='bold', color='#be185d')

    ax.text(12.6, 2.6, "Eviction Mechanism:\n• When Size >= K:\n  Evict min(S_i)\n• Low-Trust Forum Noise (tau=0.48) -> Evicted\n• High-Trust WHO Guideline (tau=0.98) -> Protected\n• Frequency Starvation Prevented via phi\n• Recency Decay via delta_t", 
            ha='center', va='center', fontsize=8.5, color='#1e293b')

    # Memory Footprint proof banner
    rect_proof = patches.FancyBboxPatch((11.0, 0.8), 3.2, 0.8, boxstyle="round,pad=0.1", 
                                       fc='#fce7f3', ec='#ec4899', lw=1.2)
    ax.add_patch(rect_proof)
    ax.text(12.6, 1.2, "RAM: 450.9 MB -> 38.3 MB\n(87.8% - 91.5% RAM Reduction)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#9d174d')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Success] Saved Architecture Diagram to: {output_path}")

if __name__ == "__main__":
    draw_architecture_diagram()
