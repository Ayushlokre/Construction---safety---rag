"""Deterministic routes for two known OSHA source visuals."""
def known_source_visual_pages(prompt: str):
    q = (prompt or "").lower()
    if "common hazards" in q and "descriptions" in q and ("3071" in q or "job hazard" in q):
        return {"title": "Common Hazards and Descriptions table",
                "source": "OSHA_3071_Job_Hazard_Analysis.pdf", "pages": [47, 48]}
    if "hierarchy of controls" in q and ("3886" in q or "construction safety" in q):
        return {"title": "Hierarchy of Controls diagram",
                "source": "OSHA_3886_Construction_Safety.pdf", "pages": [25]}
    return None
