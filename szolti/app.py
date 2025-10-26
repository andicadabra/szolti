import streamlit as st
import math
from typing import Optional

class SailRecommendation:
    def __init__(self):
        # Baseline constants chosen to preserve your original outputs at 18 kn
        self.K_male = 5.0     # baseline at 18 kn, 70 kg
        self.K_female = 4.7   # baseline at 18 kn, 60 kg
        self.p = 0.80         # weight exponent
        self.q = 1.80         # wind exponent
        self.Vref = 18.0      # knots
        self.rho0 = 1.225     # kg/m^3 (sea level, ~15°C)

        # Gentle context multipliers
        self.discipline_map = {
            "freestyle": 0.95,
            "freeride": 1.05,
            "wave": 0.95,
            "slalom": 1.07,
        }
        self.sea_map = {"flat": 1.10, "chop": 1.00, "waves": 0.95}

    # ---------- helpers ----------
    def _eff_wind(self, wind_kn: float, gust_kn: Optional[float] = None, gust_weight: float = 0.25) -> float:
        """RMS blend of mean and gust; keep gw ∈ [0, 0.4]."""
        if gust_kn is None:
            return wind_kn
        gw = max(0.0, min(0.4, float(gust_weight)))
        return math.sqrt((1.0 - gw) * wind_kn**2 + gw * gust_kn**2)

    def _air_density(self, temp_C: float = 15.0, pressure_hPa: float = 1013.25) -> float:
        """Simple ideal-gas scaling around ISA sea level."""
        T = temp_C + 273.15
        return 1.225 * (pressure_hPa / 1013.25) * (273.15 / T)

    def _context_factor(self,
                        discipline: str = "freestyle",
                        sea_state: str = "flat",
                        rig_eff: float = 1.00,
                        skill_factor: float = 1.00) -> float:
        Fd = self.discipline_map.get((discipline or "").lower(), 1.00)
        Fs = self.sea_map.get((sea_state or "").lower(), 1.00)
        return Fd * Fs * float(rig_eff) * float(skill_factor)

    def _area_core(self, base_K: float, weight_kg: float, wind_kn: float,
                   *,
                   gust_kn: Optional[float] = None,
                   temp_C: float = 15.0,
                   pressure_hPa: float = 1013.25,
                   discipline: str = "freestyle",
                   sea_state: str = "flat",
                   rig_eff: float = 1.00,
                   skill_factor: float = 1.00,
                   ref_weight: float = 70.0) -> float:
        if wind_kn <= 0:
            return float("nan")
        Veff = self._eff_wind(wind_kn, gust_kn=gust_kn)
        rho = self._air_density(temp_C=temp_C, pressure_hPa=pressure_hPa)
        Fctx = self._context_factor(discipline, sea_state, rig_eff, skill_factor)
        # A = K * (W/Wref)^p * (Vref/Veff)^q * (rho0/rho) * Fctx
        return (
            base_K
            * (max(1e-6, weight_kg) / ref_weight) ** self.p
            * (self.Vref / max(0.1, Veff)) ** self.q
            * (self.rho0 / rho)
            * Fctx
        )

    # ---------- public API (backward-compatible) ----------
    def get_sail_recommendation_for_men(self, weight: float, wind: float, skill: float, **kwargs) -> float:
        return self._area_core(self.K_male, weight, wind, skill_factor=skill, ref_weight=70.0, **kwargs)

    def get_sail_recommendation_for_women(self, weight: float, wind: float, skill: float, **kwargs) -> float:
        return self._area_core(self.K_female, weight, wind, skill_factor=skill, ref_weight=60.0, **kwargs)

    def recommend(self, sex: str, weight: float, wind: float, skill: float, **kwargs) -> float:
        s = (sex or "").strip().lower()
        if s == "male":
            return self.get_sail_recommendation_for_men(weight, wind, skill, **kwargs)
        if s == "female":
            return self.get_sail_recommendation_for_women(weight, wind, skill, **kwargs)
        raise ValueError("sex must be 'male' or 'female'")

st.set_page_config(page_title="Windsurf Sail Size Recommender", layout="centered")
st.title("🏄‍♂️ Windsurf Sail Size Recommender by Andreas Rössler")
st.caption("Estimate sail size from wind (kn), rider weight (kg), and skill factor.")

with st.sidebar:
    st.header("About")
    st.write("• Quick guide for freeride/freestyle\n"
             "• Best around 18–25 kn\n"
             "• Units: wind=knots, weight=kg, result=m²")

# Inputs (nice layout)
col1, col2, col3 = st.columns(3)
with col1:
    sex = st.radio("Rider", ["Male", "Female"], horizontal=True)
with col2:
    weight = st.number_input("Weight (kg)", min_value=25.0, max_value=150.0, value=75.0, step=1.0)
with col3:
    wind = st.number_input("Wind (kn)", min_value=1.0, max_value=60.0, value=18.0, step=1.0)

SKILL = {"Beginner": 0.9, "Intermediate": 1.0, "Advanced": 1.1}
skill_label = st.select_slider("Skill", options=list(SKILL.keys()), value="Intermediate")
skill = SKILL[skill_label]

# --- Advanced (optional) ---
with st.expander("Advanced conditions"):
    st.caption("Enable only the inputs you actually know. Disabled items are ignored.")

    # Gusts
    use_gust = st.checkbox("Use gust", value=False)
    if use_gust:
        gust_kn = st.number_input("Gust (kn)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Peak gust speed in knots")

    # Air density (temperature & pressure)
    use_air = st.checkbox("Use air temperature & pressure", value=False)
    if use_air:
        temp_C = st.number_input("Air temperature (°C)", min_value=-10.0, max_value=45.0, value=15.0, step=1.0)
        pressure_hPa = st.number_input("Air pressure (hPa)", min_value=850.0, max_value=1050.0, value=1013.0, step=1.0)

    # Discipline & sea state multipliers
    use_context = st.checkbox("Use discipline & sea state", value=False)
    if use_context:
        discipline = st.selectbox("Discipline", ["freestyle", "freeride", "wave", "slalom"], index=0)
        sea_state = st.selectbox("Sea state", ["flat", "chop", "waves"], index=0)

    # Rig efficiency multiplier
    use_rig = st.checkbox("Use rig efficiency", value=False)
    if use_rig:
        rig_eff = st.slider("Rig efficiency", min_value=0.95, max_value=1.05, value=1.00, step=0.01,
                            help="<1.00 = efficient race-y rigs; >1.00 = softer/older rigs")

if st.button("Calculate", type="primary", key="calc_main"):
    model = SailRecommendation()
    kwargs = {}
    if 'use_gust' in locals() and use_gust and 'gust_kn' in locals() and gust_kn > 0:
        kwargs["gust_kn"] = gust_kn
    if 'use_air' in locals() and use_air and 'temp_C' in locals() and 'pressure_hPa' in locals():
        kwargs["temp_C"] = temp_C
        kwargs["pressure_hPa"] = pressure_hPa
    if 'use_context' in locals() and use_context and 'discipline' in locals() and 'sea_state' in locals():
        kwargs["discipline"] = discipline
        kwargs["sea_state"] = sea_state
    if 'use_rig' in locals() and use_rig and 'rig_eff' in locals():
        kwargs["rig_eff"] = rig_eff
    size = model.recommend(sex, weight, wind, skill, **kwargs)



    if math.isnan(size):
        st.error("Wind must be greater than 0.")
    else:
        rounded = round(size, 1)
        st.metric("Recommended sail size", f"{rounded} m²", help="Rounded to 0.1 m²")
        st.write(
            f"For a **{sex.lower()}** rider at **{weight:.0f} kg** and **{wind:.0f} kn** "
            f"with skill **{skill:.1f}**, the estimate is **~{rounded} m²**."
        )
        # Nearby sizes
        cands = sorted({
            max(2.5, round(rounded - 0.5, 1)),
            round(rounded, 1),
            min(8.5, round(rounded + 0.5, 1)),
        })
        st.write("Nearby sizes you might also consider: " + ", ".join(f"{c} m²" for c in cands))

st.markdown("---\n⚠️ Always consider board/fin, gustiness, and preference.")
