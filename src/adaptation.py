"""Rule-based adaptation suggestions. General decision support, not emergency instructions
or site-specific engineering advice."""
NOTICE = ("General decision-support suggestions only. Follow official warnings from Malawi's "
          "responsible authorities; obtain site-specific engineering advice for infrastructure.")

FLOOD = ["Confirm early-warning contacts and evacuation routes with the District Civil Protection Committee.",
         "Clear and inspect drains, culverts and field outlets before heavy rain.",
         "Move equipment, inputs and stored produce above likely flood levels.",
         "Plan post-flood checks: field access, canal and embankment damage, waterlogging."]
DROUGHT = ["Review water allocation and prioritise by crop stage and water sensitivity.",
           "Conserve soil moisture (mulch/trash blanket, minimum disturbance where appropriate).",
           "Check storage and standby pumping; prepare an irrigation-failure contingency.",
           "Consider adjusted planting dates, drought-tolerant varieties and diversification next season."]
SUGARCANE = ["Schedule irrigation from measured soil moisture/ETc; avoid water stress in the grand-growth phase.",
             "Maintain canals and field drains: they serve both drought supply and flood evacuation.",
             "Plan harvest and haulage access routes for wet-weather conditions.",
             "Account for river-abstraction risk: low Shire flows can limit pumping."]

def recommend(flood_flag: bool, drought_flag: bool, sugarcane: bool = False) -> dict:
    items = []
    if flood_flag: items += [("Flood", s) for s in FLOOD]
    if drought_flag: items += [("Drought", s) for s in DROUGHT]
    if sugarcane and (flood_flag or drought_flag): items += [("Sugarcane/irrigation", s) for s in SUGARCANE]
    if flood_flag and drought_flag:
        items.append(("Compound", "Both signals present: keep drainage open while securing water storage."))
    return {"notice": NOTICE, "items": items}
