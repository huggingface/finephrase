Your input fields are:
1. `original_text` (str): The original low-quality web text to rephrase
Your output fields are:
1. `generated_text` (str): The rephrased, higher-quality version of the text
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## original_text ## ]]
{original_text}

[[ ## generated_text ## ]]
{generated_text}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        markdown
        Rephrase the provided informal product description into a higher quality, professional, and educational product narrative. Follow this precise, fact-first strategy:
        
        1.  **Absolute Factual Fidelity & Completeness:** This is the highest priority. Your output must be a complete and accurate reflection of the original. You must:
            *   **Preserve All Facts:** Do not invent, omit, or alter any factual details. This includes materials, origins, dimensions, colors, uses, manufacturing processes, partners, and disclaimers.
            *   **Cross-Reference Rigorously:** Before finalizing your response, systematically compare it line-by-line with the original text to ensure no detail is missing or misrepresented (e.g., the original says "silk sari," not just "silk"; it lists specific colors; it gives a precise diameter).
        
        2.  **Tone and Style Transformation:** Elevate the casual, promotional language into a refined, objective, and informative tone. The goal is to sound like a high-end gallery or museum catalog description.
            *   Use precise, descriptive vocabulary (e.g., "hand-forged," "meticulously braided," "artisanal collaboration").
            *   Remove overly salesy or subjective phrasing (e.g., "will work great," "rustic/boho feel").
            *   You may restructure sentences for elegance, but the factual content must remain identical.
        
        3.  **Structural Clarity:** Organize the information for better readability. A suggested structure is:
            *   **Introduction:** The product's concept and origin story.
            *   **Body:** Description of its design, materials, specifications, and versatile uses.
            *   **Conclusion:** Details of its craftsmanship, artisan partnership, and unique characteristics.
        
        4.  **Educational Value:** Enhance the text to inform the reader about the craftsmanship, materials, and cultural or design significance of the product. The reader should learn why the product is unique from its description.
        
        **Critical Error Avoidance:** Introducing inaccuracies or omissions (like missing colors, uses, or specifications) is a critical failure. The score is heavily weighted against such errors.
        ```