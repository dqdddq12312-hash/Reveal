# esg_metadata.py
"""
ESG Metadata Module for HKEx-style ESG indicators, designed following
the ESGReveal paper structure:

- Entity:      <Aspect, KPI, Topic, Quantity>
- Extensions:  SearchTerm, Knowledge

Three pillars:
    - environment (A1–A4)
    - social (B1–B8)
    - governance (G1–G11, custom but aligned with HKEx CG Code themes)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Iterable, Optional, Tuple, Any
from pathlib import Path
import json
import hashlib


# ----------------------------------------------------------------------------
# Schema metadata + changelog
# ----------------------------------------------------------------------------

METADATA_SCHEMA_VERSION = "1.1.0"
METADATA_CHANGELOG: Dict[str, List[str]] = {
    "1.1.0": [
        "Added indicator_id, evidence typing, retrieval hints, priority scoring, and dependency metadata.",
        "Introduced schema validation helpers and QA hooks for numeric/time-series expectations.",
    ],
    "1.0.0": ["Initial HKEx metadata set aligned with ESGReveal structure."],
}


# ----------------------------------------------------------------------
# Dataclass for a single metadata row (one ESG indicator)
# ----------------------------------------------------------------------


@dataclass
class ESGMetadataRecord:
    aspect: str
    kpi: str
    topic: List[str]
    quantity: str          # "Absolute Values" | "Key Actions"
    search_terms: List[str]
    knowledge: str
    indicator_id: Optional[str] = None
    evidence_type: str = "auto"  # auto | quantitative | qualitative | policy
    regex_hints: List[str] = field(default_factory=list)
    multilingual_synonyms: List[str] = field(default_factory=list)
    canonical_units: List[str] = field(default_factory=list)
    priority: str = "medium"  # low | medium | high | critical
    dependency_groups: List[str] = field(default_factory=list)
    expected_year_span: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.indicator_id:
            slug = self._slugify_aspect()
            digest = hashlib.sha1(f"{self.aspect}|{self.kpi}".encode("utf-8")).hexdigest()[:8]
            self.indicator_id = f"{slug}-{digest}"
        self.evidence_type = self._resolve_evidence_type(self.evidence_type)
        self.priority = self._normalise_priority(self.priority)
        self._ensure_canonical_units()

    def _slugify_aspect(self) -> str:
        return self.aspect.split()[0].replace(".", "").upper()

    def _resolve_evidence_type(self, value: str) -> str:
        normalised = (value or "auto").strip().lower()
        if normalised in {"quantitative", "qualitative", "policy"}:
            return normalised
        if normalised == "auto":
            return "quantitative" if self.quantity.lower() == "absolute values" else "qualitative"
        return "qualitative"

    @staticmethod
    def _normalise_priority(value: str) -> str:
        value = (value or "medium").strip().lower()
        return value if value in {"low", "medium", "high", "critical"} else "medium"

    def _ensure_canonical_units(self) -> None:
        if self.canonical_units or not self.requires_numeric_evidence():
            return
        units = self._infer_canonical_units()
        if units:
            self.canonical_units = units

    def _infer_canonical_units(self) -> List[str]:
        text = self.kpi.lower()
        if "emission" in text:
            if "scope" in text or "greenhouse" in text:
                return ["tonnes CO2e", "tonnes of greenhouse gas"]
            return ["tonnes"]
        if "waste" in text:
            return ["tonnes"]
        if "energy" in text:
            return ["kWh", "MWh"]
        if "water" in text or "wastewater" in text:
            return ["cubic metres (m3)"]
        if "packaging" in text or "material" in text:
            return ["tonnes"]
        if "headcount" in text or "number of employees" in text:
            return ["people"]
        if "turnover" in text and "rate" in text:
            return ["percent"]
        if "training" in text and "hours" in text:
            return ["hours"]
        if "fatalit" in text or "injur" in text:
            return ["cases", "rate per 1000 employees"]
        if "community" in text and ("donation" in text or "resources" in text):
            return ["HKD", "USD"]
        return []

    # ---- Helper hooks -----------------------------------------------------

    def requires_numeric_evidence(self) -> bool:
        return self.quantity.lower() == "absolute values" or self.evidence_type == "quantitative"

    def requires_time_series(self) -> bool:
        return bool(self.expected_year_span)

    def expected_time_series(self) -> Optional[int]:
        return self.expected_year_span

    def retrieval_hints(self) -> Dict[str, List[str]]:
        return {
            "search_terms": self.search_terms,
            "regex": self.regex_hints,
            "synonyms": self.multilingual_synonyms,
            "canonical_units": self.canonical_units,
        }

    def to_entity_tuple(self) -> Tuple[str, str, List[str], str]:
        """Return the <Aspect, KPI, Topic, Quantity> tuple."""
        return (self.aspect, self.kpi, self.topic, self.quantity)

    def to_json(self) -> Dict:
        """JSON‑serialisable dict representation."""
        return asdict(self)


def validate_metadata(metadata: Dict[str, List[ESGMetadataRecord]]) -> List[str]:
    """Validate metadata schema completeness; returns list of human-readable errors."""
    errors: List[str] = []
    seen_ids: Dict[str, str] = {}
    for pillar, records in metadata.items():
        for record in records:
            if not record.topic:
                errors.append(f"[{record.indicator_id}] missing topic definitions ({pillar}).")
            if not record.search_terms:
                errors.append(f"[{record.indicator_id}] missing search_terms ({pillar}).")
            if record.indicator_id in seen_ids and seen_ids[record.indicator_id] != record.kpi:
                errors.append(f"Duplicate indicator_id detected: {record.indicator_id}.")
            seen_ids[record.indicator_id] = record.kpi
    return errors


# ----------------------------------------------------------------------
# Raw metadata definition for all pillars
# ----------------------------------------------------------------------

ESG_METADATA: Dict[str, List[ESGMetadataRecord]] = {
    # =========================
    # A. ENVIRONMENTAL (A1–A4)
    # =========================
    "environment": [
        # ----- A1: Emissions -----
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Policies, governance and compliance related to air emissions, discharges into water and land, and hazardous waste handling.",
            topic=[
                "Environmental policy", "Air and GHG compliance", "Wastewater governance", "Hazardous waste policy",
            ],
            quantity="Key Actions",
            search_terms=[
                "environmental policy", "emissions policy", "wastewater compliance", "hazardous waste policy",
                "pollution control governance", "environmental management system",
            ],
            knowledge=(
                "HKEx requires issuers to describe overarching policies, management systems and compliance records for air emissions, "
                "effluent discharges and hazardous waste. Summaries should cover governance roles, applicable regulations and any "
                "cases of non-compliance or corrective actions."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Types of key air emissions (e.g. NOx, SOx, particulates) and the related emission data.",
            topic=["Air emissions", "Nitrogen oxides (NOx)", "Sulfur oxides (SOx)", "Particulate matter (PM)"],
            quantity="Absolute Values",
            search_terms=[
                "air emissions", "NOx", "nitrogen oxides", "SOx", "sulfur oxides",
                "sulphur dioxide", "particulate matter", "PM", "stack emission",
            ],
            knowledge=(
                "This indicator covers regulated air pollutants such as NOx, SOx and particulate "
                "matter. Issuers typically report total emissions and, where useful, intensity "
                "figures (for example per unit of production)."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Direct (Scope 1) and energy‑indirect (Scope 2) greenhouse gas emissions, together with intensity indicators.",
            topic=["Scope 1 emissions", "Scope 2 emissions", "Greenhouse gas intensity"],
            quantity="Absolute Values",
            search_terms=[
                "greenhouse gas", "GHG", "carbon footprint", "CO2e", "scope 1", "scope 2",
                "GHG emissions", "GHG intensity",
            ],
            knowledge=(
                "Scope 1 refers to direct greenhouse‑gas emissions from owned or controlled "
                "sources. Scope 2 covers indirect emissions from purchased electricity, steam, "
                "heating or cooling. Disclosure usually includes tonnes of CO2‑equivalent and "
                "intensity per activity unit."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Scope 1 greenhouse gas emissions by major source (fuel combustion, process emissions) and related intensity.",
            topic=["Scope 1 breakdown", "Stationary combustion", "Mobile combustion", "GHG intensity"],
            quantity="Absolute Values",
            search_terms=[
                "scope 1 emissions", "direct emissions", "stationary combustion", "mobile combustion",
                "process emissions", "scope 1 intensity",
            ],
            knowledge=(
                "HKEx indicators expect issuers to quantify direct (Scope 1) emissions by key "
                "sources to show which operations drive the footprint. Typical disclosures split "
                "fuel combustion, process gases and vehicles together with an intensity ratio."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Scope 2 greenhouse gas emissions, separated into location-based and market-based values where available, plus intensity.",
            topic=["Scope 2 location-based", "Scope 2 market-based", "Purchased electricity", "GHG intensity"],
            quantity="Absolute Values",
            search_terms=[
                "scope 2 emissions", "purchased electricity", "location-based", "market-based",
                "indirect emissions", "scope 2 intensity",
            ],
            knowledge=(
                "Energy-indirect emissions (Scope 2) should be broken into location-based and "
                "market-based values when renewable instruments are used. Intensities such as "
                "tonnes CO2e per MWh or per unit of revenue help show efficiency improvements."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Total hazardous waste generated and the corresponding intensity per activity unit.",
            topic=["Hazardous waste volume", "Hazardous waste intensity"],
            quantity="Absolute Values",
            search_terms=[
                "hazardous waste", "chemical waste", "dangerous waste", "toxic waste",
                "tonnes of hazardous waste", "hazardous waste intensity",
            ],
            knowledge=(
                "Hazardous waste includes wastes classified as hazardous by regulation, such as "
                "solvents, oils, batteries and contaminated materials. Companies normally report "
                "total tonnes and an intensity metric (for example per unit of output)."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Total non‑hazardous waste generated and the corresponding intensity per activity unit.",
            topic=["Non‑hazardous waste volume", "Non‑hazardous waste intensity"],
            quantity="Absolute Values",
            search_terms=[
                "non-hazardous waste", "general waste", "municipal waste",
                "solid waste", "landfill", "recyclable waste", "waste intensity",
            ],
            knowledge=(
                "Non‑hazardous waste covers municipal and production wastes that are not classified "
                "as hazardous, such as paper, packaging, food waste or construction debris."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Total wastewater and effluent discharged and, where available, pollutant concentrations or intensity per facility.",
            topic=["Wastewater discharge", "Effluent pollutants", "Water quality compliance"],
            quantity="Absolute Values",
            search_terms=[
                "wastewater discharge", "effluent volume", "treated effluent", "COD", "BOD",
                "wastewater intensity", "sewage", "discharge to water",
            ],
            knowledge=(
                "HKEx highlights discharges into water and land within Aspect A1. Companies "
                "typically report wastewater volumes, pollutant loads such as COD/BOD, and any "
                "intensity indicators or compliance data tied to discharge permits."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Emission‑reduction objectives set by the company and progress or actions taken to meet those objectives.",
            topic=["Emission reduction targets", "Greenhouse‑gas targets", "Air‑pollutant targets"],
            quantity="Key Actions",
            search_terms=[
                "emission reduction target", "carbon reduction", "net-zero", "science-based target",
                "emission roadmap", "implementation plan",
            ],
            knowledge=(
                "Focuses on whether the issuer sets numerical or qualitative goals for reducing "
                "emissions and describes plans, milestones and performance against those goals."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Methods for handling hazardous and non‑hazardous waste, together with reduction targets and measures implemented.",
            topic=["Waste management practices", "Hazardous waste handling",
                   "Non‑hazardous waste handling", "Waste reduction targets"],
            quantity="Key Actions",
            search_terms=[
                "waste management", "waste segregation", "recycling programme",
                "safe disposal", "waste reduction target", "waste minimisation",
            ],
            knowledge=(
                "Covers policies and procedures for collection, segregation, recycling and disposal "
                "of different waste streams, as well as initiatives and targets to reduce waste."
            ),
        ),
        ESGMetadataRecord(
            aspect="A1. Emissions",
            kpi="Processes for wastewater treatment, spill prevention and soil contamination control, including monitoring and response actions.",
            topic=["Wastewater treatment", "Spill prevention", "Soil protection", "Monitoring"],
            quantity="Key Actions",
            search_terms=[
                "wastewater treatment", "effluent control", "spill response", "bunding", "secondary containment",
                "soil remediation", "pollution incident", "environmental monitoring",
            ],
            knowledge=(
                "Aspect A1 requires issuers to explain how they prevent and respond to discharges "
                "to water and land. Useful details include treatment technologies, secondary "
                "containment, spill drills, incident statistics and remediation steps."
            ),
        ),

        # ----- A2: Use of Resources -----
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Policies and governance for efficient use of energy, water, raw materials and packaging, including compliance with relevant regulations.",
            topic=["Resource efficiency policy", "Energy governance", "Water stewardship", "Materials management"],
            quantity="Key Actions",
            search_terms=[
                "resource efficiency policy", "energy management policy", "water policy",
                "materials management", "ISO 50001", "ISO 14001",
            ],
            knowledge=(
                "Under Aspect A2, HKEx expects issuers to outline policies, responsibilities and "
                "compliance processes for conserving energy, water and materials. Mention targets, "
                "management systems and any regulatory obligations."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Total direct and indirect energy consumption by type and energy intensity.",
            topic=["Energy consumption", "Energy intensity", "Electricity use", "Fuel use"],
            quantity="Absolute Values",
            search_terms=[
                "energy consumption", "electricity consumption", "gas consumption",
                "fuel consumption", "kWh", "energy intensity",
            ],
            knowledge=(
                "Reports total energy use split by energy sources together with an intensity metric, "
                "for example kilowatt‑hours per unit of production or per floor area."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Renewable energy consumption (self-generated and procured) and its share of total energy use.",
            topic=["Renewable electricity", "Self-generated energy", "Green purchase", "Energy mix"],
            quantity="Absolute Values",
            search_terms=[
                "renewable energy", "green electricity", "RECs", "solar generation",
                "renewable consumption", "percentage renewable",
            ],
            knowledge=(
                "To evidence decarbonisation roadmaps, issuers often disclose megawatt-hours of "
                "renewable energy consumed (onsite or procured) together with the proportion of total "
                "energy use covered."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Renewable energy sourcing strategy, including procurement instruments, onsite projects and future targets.",
            topic=["Renewable strategy", "Power purchase", "Onsite solar", "Renewable targets"],
            quantity="Key Actions",
            search_terms=[
                "renewable energy strategy", "PPA", "onsite solar", "renewable roadmap",
                "green electricity plan", "science based target",
            ],
            knowledge=(
                "HKEx encourages issuers to explain how they increase the share of renewable "
                "energy, for example via PPAs, rooftop solar, renewable certificates or supplier "
                "engagement, together with milestones and investments."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Total water consumption and water‑use intensity.",
            topic=["Water consumption", "Water intensity"],
            quantity="Absolute Values",
            search_terms=[
                "water use", "water consumption", "freshwater", "m3 of water",
                "water intensity", "potable water",
            ],
            knowledge=(
                "Includes overall water withdrawal and an intensity indicator such as cubic metres "
                "per unit of production, per employee or per square metre of floor space."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Targets for improving energy efficiency and actions taken to achieve them.",
            topic=["Energy efficiency targets", "Energy‑saving measures"],
            quantity="Key Actions",
            search_terms=[
                "energy efficiency target", "energy saving plan", "energy conservation",
                "equipment upgrade", "energy management system",
            ],
            knowledge=(
                "Looks at whether the issuer sets goals to improve energy efficiency and describes "
                "initiatives such as equipment upgrades, process optimisation or controls."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Issues in sourcing water fit for use, and any water‑efficiency targets and measures.",
            topic=["Water sourcing issues", "Water efficiency targets"],
            quantity="Key Actions",
            search_terms=[
                "water scarcity", "water stress", "water sourcing risk",
                "water efficiency target", "water-saving measures",
            ],
            knowledge=(
                "Covers risks or constraints in obtaining water that meets quality requirements, "
                "and programmes to improve water efficiency and secure supply."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Consumption of key raw or packaging materials (e.g. metals, plastics, paper) and intensity per unit of output.",
            topic=["Raw material use", "Material intensity", "Plastics", "Metals"],
            quantity="Absolute Values",
            search_terms=[
                "raw material consumption", "plastic use", "steel consumption", "materials per unit",
                "material intensity", "GRI 301",
            ],
            knowledge=(
                "Aspect A2 links to GRI 301 on materials. Disclose tonnage of major inputs such as "
                "metals, plastics, paper or bio-based feedstock plus a normalised intensity metric."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Total packaging materials used for finished products and, where relevant, packaging intensity.",
            topic=["Packaging material use", "Packaging intensity"],
            quantity="Absolute Values",
            search_terms=[
                "packaging material", "paper packaging", "plastic packaging",
                "packaging tonnes", "packaging per unit",
            ],
            knowledge=(
                "Reports the amount of packaging materials such as paper, plastics or metals used "
                "for finished goods, often with a metric per unit of product."
            ),
        ),
        ESGMetadataRecord(
            aspect="A2. Use of Resources",
            kpi="Initiatives to reduce packaging impact, increase recycled content or eliminate single-use materials, and results achieved.",
            topic=["Circular packaging", "Recycled content", "Packaging reduction", "Eco-design"],
            quantity="Key Actions",
            search_terms=[
                "recycled content", "plastic reduction", "packaging redesign", "circular packaging",
                "eliminate single-use", "packaging roadmap",
            ],
            knowledge=(
                "HKEx references product lifecycle considerations under A2. Describe packaging "
                "redesigns, recycled input targets, refill models or supplier requirements together "
                "with quantitative results where possible."
            ),
        ),

        # ----- A3: Environment and Natural Resources -----
        ESGMetadataRecord(
            aspect="A3. Environment and Natural Resources",
            kpi="Policies and governance processes for minimising significant impacts on the environment and natural resources.",
            topic=["Impact management", "Environmental governance", "Resource conservation policy"],
            quantity="Key Actions",
            search_terms=[
                "natural resource policy", "biodiversity policy", "environmental impact management",
                "restoration standard", "no net loss",
            ],
            knowledge=(
                "Aspect A3 general disclosure expects companies to describe how they identify "
                "significant impacts on ecosystems or natural resources, assign responsibilities and "
                "embed mitigation into operations and capital projects."
            ),
        ),
        ESGMetadataRecord(
            aspect="A3. Environment and Natural Resources",
            kpi="Significant impacts of the company’s activities on the environment and natural resources, and actions taken to manage them.",
            topic=["Significant environmental impacts", "Natural resource impacts", "Mitigation actions"],
            quantity="Key Actions",
            search_terms=[
                "biodiversity impact", "deforestation", "resource depletion",
                "environmental impact assessment", "mitigation measures",
            ],
            knowledge=(
                "Addresses how operations affect ecosystems and natural resources, and what "
                "management actions or restoration efforts are in place."
            ),
        ),
        ESGMetadataRecord(
            aspect="A3. Environment and Natural Resources",
            kpi="Biodiversity conservation or habitat restoration programmes, offsets and partnerships, including progress metrics.",
            topic=["Biodiversity", "Habitat restoration", "Offsets", "Partnerships"],
            quantity="Key Actions",
            search_terms=[
                "biodiversity program", "habitat restoration", "ecological offset", "conservation partnership",
                "tree planting", "nature positive",
            ],
            knowledge=(
                "Companies with land-intensive operations should disclose dedicated initiatives to "
                "protect or restore habitats, such as reforestation, offset programmes or work with "
                "NGOs, and report qualitative or quantitative progress."
            ),
        ),

        # ----- A4: Climate Change -----
        ESGMetadataRecord(
            aspect="A4. Climate Change",
            kpi="Policies, governance arrangements and compliance processes for identifying and managing climate-related risks and opportunities.",
            topic=["Climate governance", "Risk oversight", "Compliance", "Scenario governance"],
            quantity="Key Actions",
            search_terms=[
                "climate governance", "board oversight climate", "TCFD governance", "climate policy",
                "climate risk management", "scenario analysis",
            ],
            knowledge=(
                "HKEx aligns Aspect A4 with TCFD concepts. Issuers should outline governance "
                "structures, management responsibilities, compliance processes and how climate "
                "risks are integrated into enterprise risk management."
            ),
        ),
        ESGMetadataRecord(
            aspect="A4. Climate Change",
            kpi="Major climate‑related issues that affect or could affect the company, and actions taken to manage them.",
            topic=["Climate risks", "Climate opportunities", "Adaptation and mitigation actions"],
            quantity="Key Actions",
            search_terms=[
                "physical climate risk", "transition risk", "climate scenario",
                "adaptation measure", "mitigation action", "TCFD",
            ],
            knowledge=(
                "Covers governance and management of climate‑related risks and opportunities, "
                "including how the business adapts to and reduces climate impacts."
            ),
        ),
    ],

    # =========================
    # B. SOCIAL (B1–B8)
    # =========================
    "social": [
        # ----- B1: Employment -----
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Policies, governance arrangements and compliance with laws on compensation, recruitment, working hours and equal opportunity.",
            topic=["Employment policy", "Compensation governance", "Working hours", "Equal opportunity"],
            quantity="Key Actions",
            search_terms=[
                "employment policy", "compensation policy", "working hours", "rest day",
                "equal opportunity policy", "non-discrimination", "employment compliance",
            ],
            knowledge=(
                "Aspect B1 requires issuers to explain employment-related policies, board oversight "
                "and compliance with Hong Kong labour ordinances, including diversity and benefits."
            ),
        ),
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Total workforce, broken down by gender, employment type, age group and geographic region.",
            topic=["Workforce size", "Gender distribution", "Employment type", "Age group", "Region"],
            quantity="Absolute Values",
            search_terms=[
                "headcount", "number of employees", "full-time", "part-time",
                "gender breakdown", "age group", "region",
            ],
            knowledge=(
                "Requires a structured breakdown of the workforce, typically in tables, showing how "
                "employees are distributed by gender, contract type, age band and location."
            ),
        ),
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Workforce by employment type (full-time, part-time, contract) with totals and percentages.",
            topic=["Full-time", "Part-time", "Contract", "Headcount"],
            quantity="Absolute Values",
            search_terms=[
                "full-time employees", "part-time employees", "contract staff", "headcount by employment type",
                "permanent employees", "temporary employees",
            ],
            knowledge=(
                "HKEx KPI B1.1 explicitly calls for employment type splits. Extract headcount and, "
                "if provided, the percentage of the workforce in each contract type."
            ),
        ),
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Employee headcount by age group (e.g. under 30, 30–50, over 50) and corresponding percentages.",
            topic=["Age under 30", "Age 30-50", "Age over 50", "Age distribution"],
            quantity="Absolute Values",
            search_terms=[
                "age group", "under 30", "30-50", "over 50", "age distribution", "demographics",
            ],
            knowledge=(
                "Provide the number of employees in each disclosed age band together with any "
                "percentage metrics required by KPI B1.1."
            ),
        ),
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Employee headcount by geographic region (e.g. Hong Kong, Mainland China, Overseas) and related percentages.",
            topic=["Region", "Hong Kong", "Mainland China", "Overseas"],
            quantity="Absolute Values",
            search_terms=[
                "employees in Hong Kong", "employees in mainland", "regional headcount", "geographic distribution",
            ],
            knowledge=(
                "Capture the workforce split by key geographies highlighted in ESG reports to satisfy "
                "the regional breakdown element of KPI B1.1."
            ),
        ),
        ESGMetadataRecord(
            aspect="B1. Employment",
            kpi="Employee turnover rate by gender, age group and geographic region.",
            topic=["Turnover rate", "Gender", "Age group", "Region"],
            quantity="Absolute Values",
            search_terms=[
                "employee turnover", "staff turnover rate", "resignation rate",
                "attrition", "voluntary turnover", "involuntary turnover",
            ],
            knowledge=(
                "Focuses on how frequently employees leave the company, segmented by demographic "
                "categories, usually expressed as a percentage over the year."
            ),
        ),

        # ----- B2: Health and Safety -----
        ESGMetadataRecord(
            aspect="B2. Health and Safety",
            kpi="Policies and compliance processes for occupational health and safety, including governance and monitoring systems.",
            topic=["Safety policy", "Compliance", "Governance", "Monitoring"],
            quantity="Key Actions",
            search_terms=[
                "health and safety policy", "OHS governance", "safety compliance",
                "safety committee", "occupational health programme",
            ],
            knowledge=(
                "Aspect B2 calls for disclosures on OHS policies, responsibilities, compliance with "
                "laws and how effectiveness is monitored across operations and contractors."
            ),
        ),
        ESGMetadataRecord(
            aspect="B2. Health and Safety",
            kpi="Number and rate of work‑related fatalities in each of the past three years, including the reporting year.",
            topic=["Work-related fatalities", "Fatality rate"],
            quantity="Absolute Values",
            search_terms=[
                "work-related fatality", "fatal accident", "occupational death",
                "fatality rate", "lost time injury fatality",
            ],
            knowledge=(
                "Captures the most serious health and safety outcomes, requiring disclosure of both "
                "the count of fatalities and related rates over a three‑year period."
            ),
        ),
        ESGMetadataRecord(
            aspect="B2. Health and Safety",
            kpi="Total lost days due to work injuries.",
            topic=["Lost days", "Work injuries"],
            quantity="Absolute Values",
            search_terms=[
                "lost work days", "days lost due to injury",
                "occupational injury days", "LTI days",
            ],
            knowledge=(
                "Measures the impact of non‑fatal workplace injuries on productivity by reporting the "
                "number of working days lost."
            ),
        ),
        ESGMetadataRecord(
            aspect="B2. Health and Safety",
            kpi="Number of recordable work-related injuries and associated injury rates (e.g. LTIR, TRIR).",
            topic=["Recordable injuries", "Injury rate", "LTIR", "TRIR"],
            quantity="Absolute Values",
            search_terms=[
                "recordable injuries", "injury rate", "LTIR", "TRIR", "lost-time injury rate",
                "OSHA rate", "work injury cases",
            ],
            knowledge=(
                "Although KPI B2.1 emphasises fatalities, issuers typically also disclose non-fatal "
                "injury counts and rates to demonstrate safety performance trends."
            ),
        ),
        ESGMetadataRecord(
            aspect="B2. Health and Safety",
            kpi="Occupational health and safety measures adopted and how these measures are implemented and monitored.",
            topic=["Health and safety policies", "Safety programmes", "Monitoring mechanisms"],
            quantity="Key Actions",
            search_terms=[
                "safety management system", "OHSAS", "ISO 45001",
                "safety training", "risk assessment", "incident investigation",
            ],
            knowledge=(
                "Describes the systems, procedures and training used to prevent workplace injuries "
                "and illnesses, and how compliance and effectiveness are monitored."
            ),
        ),

        # ----- B3: Development and Training -----
        ESGMetadataRecord(
            aspect="B3. Development and Training",
            kpi="Policies for improving employee skills, training governance and how programmes are evaluated.",
            topic=["Training policy", "Skills development", "Programme governance"],
            quantity="Key Actions",
            search_terms=[
                "training policy", "learning governance", "talent development strategy",
                "succession planning", "skills roadmap",
            ],
            knowledge=(
                "The B3 general disclosure asks issuers to describe training objectives, coverage, "
                "budgeting and evaluation methods, highlighting links to career development."
            ),
        ),
        ESGMetadataRecord(
            aspect="B3. Development and Training",
            kpi="Percentage of employees who received training, by gender and employee category.",
            topic=["Training coverage", "Gender", "Employee category"],
            quantity="Absolute Values",
            search_terms=[
                "training coverage", "percentage of employees trained",
                "training participation", "training by gender",
            ],
            knowledge=(
                "Shows how widely training opportunities are provided across the workforce, segmented "
                "by gender and job level."
            ),
        ),
        ESGMetadataRecord(
            aspect="B3. Development and Training",
            kpi="Average training hours completed per employee, by gender and employee category.",
            topic=["Average training hours", "Gender", "Employee category"],
            quantity="Absolute Values",
            search_terms=[
                "average training hours", "training hours per employee",
                "learning hours", "training intensity",
            ],
            knowledge=(
                "Provides insight into the depth of training by indicating how many hours of learning "
                "on average each employee receives."
            ),
        ),

        # ----- B4: Labour Standards -----
        ESGMetadataRecord(
            aspect="B4. Labour Standards",
            kpi="Policies and compliance controls to prevent child labour and forced labour across operations and the supply chain.",
            topic=["Child labour policy", "Forced labour policy", "Compliance"],
            quantity="Key Actions",
            search_terms=[
                "child labour policy", "forced labour policy", "human rights governance",
                "modern slavery", "supplier human rights",
            ],
            knowledge=(
                "HKEx expects issuers to outline governance over child and forced labour risks, "
                "including due diligence scope, contractual requirements and remediation pathways."
            ),
        ),
        ESGMetadataRecord(
            aspect="B4. Labour Standards",
            kpi="Measures to review employment practices to prevent child labour and forced labour.",
            topic=["Child labour prevention", "Forced labour prevention"],
            quantity="Key Actions",
            search_terms=[
                "child labour", "forced labour", "labour standards review",
                "human rights policy", "supplier code of conduct",
            ],
            knowledge=(
                "Describes how the company assesses and updates its employment practices and supply "
                "chain standards to avoid child and forced labour."
            ),
        ),
        ESGMetadataRecord(
            aspect="B4. Labour Standards",
            kpi="Steps taken to eliminate child or forced labour practices when they are identified.",
            topic=["Corrective actions", "Remediation for labour violations"],
            quantity="Key Actions",
            search_terms=[
                "remediation plan", "corrective action", "termination of supplier",
                "worker protection", "grievance mechanism",
            ],
            knowledge=(
                "Focuses on remediation: actions taken once violations are found, such as correcting "
                "conditions, compensating affected workers or adjusting supplier relationships."
            ),
        ),

        # ----- B5: Supply Chain Management -----
        ESGMetadataRecord(
            aspect="B5. Supply Chain Management",
            kpi="Number of suppliers, broken down by geographic region.",
            topic=["Supplier count", "Supplier regions"],
            quantity="Absolute Values",
            search_terms=[
                "number of suppliers", "supplier by region",
                "supply base", "vendor locations",
            ],
            knowledge=(
                "Provides a quantitative overview of the supply base and how it is distributed across "
                "different regions."
            ),
        ),
        ESGMetadataRecord(
            aspect="B5. Supply Chain Management",
            kpi="Number of suppliers assessed or audited for environmental and social risks during the year, and coverage percentage.",
            topic=["Supplier assessments", "Audited suppliers", "Coverage"],
            quantity="Absolute Values",
            search_terms=[
                "suppliers assessed", "supplier audits", "ESG assessment coverage", "supplier screening",
            ],
            knowledge=(
                "KPI B5.2 requests issuers to indicate how many suppliers are subject to ESG "
                "engagement or assessments. Capture counts and any coverage percentages reported."
            ),
        ),
        ESGMetadataRecord(
            aspect="B5. Supply Chain Management",
            kpi="Practices for engaging suppliers, the number of suppliers subject to these practices, and how they are implemented and monitored.",
            topic=["Supplier engagement practices", "Implementation and monitoring"],
            quantity="Key Actions",
            search_terms=[
                "supplier assessment", "supplier onboarding", "supplier audit",
                "ESG criteria", "code of conduct for suppliers",
            ],
            knowledge=(
                "Describes policies and processes for selecting and managing suppliers, including "
                "ESG requirements and monitoring mechanisms."
            ),
        ),
        ESGMetadataRecord(
            aspect="B5. Supply Chain Management",
            kpi="Practices used to identify environmental and social risks along the supply chain and how these practices are implemented and monitored.",
            topic=["Supply chain environmental risk", "Supply chain social risk"],
            quantity="Key Actions",
            search_terms=[
                "supply chain risk assessment", "ESG risk screening",
                "supplier audit", "high-risk suppliers",
            ],
            knowledge=(
                "Explains how the issuer identifies and manages environmental and social risks in the "
                "supply chain, including risk-based screening or audits."
            ),
        ),
        ESGMetadataRecord(
            aspect="B5. Supply Chain Management",
            kpi="Practices used to promote environmentally preferable products and services when selecting suppliers and how they are implemented and monitored.",
            topic=["Green procurement", "Sustainable sourcing"],
            quantity="Key Actions",
            search_terms=[
                "green procurement", "sustainable sourcing", "eco-friendly products",
                "environmental criteria for suppliers",
            ],
            knowledge=(
                "Covers procurement policies that prioritise environmentally preferable inputs, such "
                "as low‑carbon materials or certified sustainable products."
            ),
        ),

        # ----- B6: Product Responsibility -----
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Percentage of total products sold or shipped that were subject to recalls for safety and health reasons.",
            topic=["Product recalls", "Recall percentage"],
            quantity="Absolute Values",
            search_terms=[
                "product recall", "safety recall", "health-related recall",
                "percentage of products recalled",
            ],
            knowledge=(
                "Indicates the scale of product safety or health issues by reporting how much of the "
                "company’s output had to be recalled."
            ),
        ),
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Number of product and service‑related complaints received and how these complaints were resolved.",
            topic=["Customer complaints", "Complaint handling"],
            quantity="Absolute Values",
            search_terms=[
                "customer complaint", "service complaint", "complaint handling",
                "customer satisfaction", "hotline",
            ],
            knowledge=(
                "Requires disclosure of complaint volumes and a description of mechanisms for handling "
                "and resolving them."
            ),
        ),
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Number of substantiated customer data privacy or cybersecurity incidents and individuals affected.",
            topic=["Data breaches", "Privacy incidents", "Individuals affected"],
            quantity="Absolute Values",
            search_terms=[
                "data breach", "privacy incident", "cybersecurity incident", "customers affected",
                "PDPO", "personal data leak",
            ],
            knowledge=(
                "GRI 418 and HKEx Aspect B6 expect issuers to quantify confirmed data privacy "
                "incidents, including the number of cases and individuals impacted."
            ),
        ),
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Practices for observing and protecting intellectual property rights.",
            topic=["Intellectual property protection"],
            quantity="Key Actions",
            search_terms=[
                "intellectual property", "IP protection", "patent", "trademark",
                "copyright", "licensing",
            ],
            knowledge=(
                "Describes how the company safeguards its own intellectual property and respects "
                "others’ rights, including relevant policies and procedures."
            ),
        ),
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Quality assurance processes and recall procedures for products and services.",
            topic=["Quality assurance", "Recall procedures"],
            quantity="Key Actions",
            search_terms=[
                "quality management system", "ISO 9001", "product testing",
                "recall procedure", "corrective action",
            ],
            knowledge=(
                "Explains how quality is controlled during design, production and service delivery, and "
                "how recalls are triggered and executed when needed."
            ),
        ),
        ESGMetadataRecord(
            aspect="B6. Product Responsibility",
            kpi="Policies on consumer data protection and privacy and how they are implemented and monitored.",
            topic=["Data protection", "Customer privacy"],
            quantity="Key Actions",
            search_terms=[
                "data privacy", "personal data protection", "cybersecurity",
                "GDPR", "PDPO", "privacy policy",
            ],
            knowledge=(
                "Covers governance and controls around the collection, storage and use of customer "
                "data, including policies, training and monitoring."
            ),
        ),

        # ----- B7: Anti-corruption -----
        ESGMetadataRecord(
            aspect="B7. Anti-corruption",
            kpi="Number of concluded legal cases involving corrupt practices against the company or its employees during the reporting period and the outcomes.",
            topic=["Corruption cases", "Case outcomes"],
            quantity="Absolute Values",
            search_terms=[
                "corruption case", "bribery case", "fraud case",
                "legal proceedings", "case outcome",
            ],
            knowledge=(
                "Provides transparency on corruption‑related legal cases, indicating both their number "
                "and whether they were upheld, dismissed or resolved."
            ),
        ),
        ESGMetadataRecord(
            aspect="B7. Anti-corruption",
            kpi="Preventive measures and whistle‑blowing mechanisms, and how these are implemented and monitored.",
            topic=["Anti‑corruption controls", "Whistle‑blowing procedures"],
            quantity="Key Actions",
            search_terms=[
                "anti-corruption policy", "code of conduct", "whistleblowing",
                "hotline", "internal control", "fraud prevention",
            ],
            knowledge=(
                "Describes policies, training, reporting channels and investigation processes designed "
                "to prevent, detect and address bribery and other corrupt practices."
            ),
        ),
        ESGMetadataRecord(
            aspect="B7. Anti-corruption",
            kpi="Anti‑corruption training provided to directors and staff.",
            topic=["Anti‑corruption training", "Training coverage"],
            quantity="Absolute Values",
            search_terms=[
                "anti-corruption training", "ethics training", "compliance training",
                "training hours", "training attendance",
            ],
            knowledge=(
                "Covers the scope and depth of ethics and anti‑corruption training programmes for "
                "board members and employees."
            ),
        ),

        # ----- B8: Community Investment -----
        ESGMetadataRecord(
            aspect="B8. Community Investment",
            kpi="Main focus areas of community contribution, such as education, environment, health or culture.",
            topic=["Contribution focus areas"],
            quantity="Key Actions",
            search_terms=[
                "community investment", "community programmes", "charitable focus areas",
                "education support", "environmental initiative", "sports sponsorship",
            ],
            knowledge=(
                "Explains which community themes the company prioritises for support, for example "
                "education, environmental protection or local livelihoods."
            ),
        ),
        ESGMetadataRecord(
            aspect="B8. Community Investment",
            kpi="Resources contributed to community programmes, for example monetary donations or volunteering time.",
            topic=["Donations", "Volunteer time", "In‑kind support"],
            quantity="Absolute Values",
            search_terms=[
                "community spending", "charitable donation", "philanthropy",
                "volunteer hours", "community investment amount",
            ],
            knowledge=(
                "Quantifies the scale of community support, including cash contributions, employee "
                "volunteering and other in‑kind resources."
            ),
        ),
    ],

    # =========================
    # C. GOVERNANCE (custom G1–G11)
    # =========================
    "governance": [
        # Quantitative governance indicators (4)
        ESGMetadataRecord(
            aspect="G1. Board Composition and Independence",
            kpi="Number of directors on the board and percentage of independent non‑executive directors.",
            topic=["Board size", "Independent non‑executive directors", "INED ratio"],
            quantity="Absolute Values",
            search_terms=[
                "board of directors", "independent non-executive director",
                "INED", "board size", "board independence percentage",
            ],
            knowledge=(
                "Provides a basic picture of governance structure, indicating board size and how many "
                "directors are independent in line with listing requirements."
            ),
        ),
        ESGMetadataRecord(
            aspect="G2. Board Diversity",
            kpi="Number and percentage of female directors on the board.",
            topic=["Female directors", "Board gender diversity"],
            quantity="Absolute Values",
            search_terms=[
                "female director", "women on board", "gender diversity",
                "board gender ratio",
            ],
            knowledge=(
                "Shows gender diversity at the top governance level by disclosing the number and share "
                "of women on the board."
            ),
        ),
        ESGMetadataRecord(
            aspect="G3. Board Meetings and Attendance",
            kpi="Number of board meetings held during the year and the average attendance rate of directors.",
            topic=["Board meetings", "Attendance rate"],
            quantity="Absolute Values",
            search_terms=[
                "board meeting", "attendance rate", "meeting attendance",
                "number of meetings",
            ],
            knowledge=(
                "Indicates how actively the board meets and participates in governance, using both "
                "meeting counts and attendance statistics."
            ),
        ),
        ESGMetadataRecord(
            aspect="G4. Board Committees",
            kpi="Number of key board committees and meetings held by each committee during the year.",
            topic=["Audit committee", "Remuneration committee", "Nomination committee", "ESG committee"],
            quantity="Absolute Values",
            search_terms=[
                "audit committee", "remuneration committee", "nomination committee",
                "ESG committee", "committee meetings",
            ],
            knowledge=(
                "Provides insight into the governance workload and structure by reporting how many "
                "committees exist and how often they meet."
            ),
        ),

        # Textual governance indicators (7)
        ESGMetadataRecord(
            aspect="G5. Roles and Responsibilities of the Board",
            kpi="Description of the board’s overall responsibilities and the division of responsibilities between the board and management.",
            topic=["Board responsibilities", "Management delegation"],
            quantity="Key Actions",
            search_terms=[
                "board responsibility", "corporate governance framework",
                "delegation of authority", "management responsibility",
            ],
            knowledge=(
                "Explains the governance framework, including what matters are reserved for the board "
                "and how day‑to‑day management is delegated."
            ),
        ),
        ESGMetadataRecord(
            aspect="G6. Board Diversity Policy",
            kpi="Description of the board diversity policy and how it is implemented and monitored.",
            topic=["Board diversity policy", "Implementation and monitoring"],
            quantity="Key Actions",
            search_terms=[
                "board diversity policy", "diversity of the board",
                "skills mix", "experience diversity",
            ],
            knowledge=(
                "Covers the issuer’s policy on diversity in the boardroom, including aspects such as "
                "gender, age, skills and background, and how the policy is applied."
            ),
        ),
        ESGMetadataRecord(
            aspect="G7. Risk Management and Internal Control",
            kpi="Description of the risk management and internal control systems and their effectiveness.",
            topic=["Risk management framework", "Internal control", "Effectiveness review"],
            quantity="Key Actions",
            search_terms=[
                "risk management system", "internal control system",
                "risk committee", "annual review of internal control",
            ],
            knowledge=(
                "Explains how the board oversees risk, the processes for identifying and managing "
                "risks, and the board’s assessment of system effectiveness."
            ),
        ),
        ESGMetadataRecord(
            aspect="G8. ESG Governance and Oversight",
            kpi="Description of the governance structure for ESG matters, including the board’s role in oversight.",
            topic=["ESG governance structure", "Board oversight of ESG"],
            quantity="Key Actions",
            search_terms=[
                "ESG committee", "sustainability committee", "board oversight of ESG",
                "ESG governance framework",
            ],
            knowledge=(
                "Describes how ESG responsibilities are allocated among the board, committees and "
                "management, and how ESG issues are reported and monitored."
            ),
        ),
        ESGMetadataRecord(
            aspect="G9. Remuneration Governance",
            kpi="Description of the remuneration policy for directors and senior management and how it aligns with long‑term performance.",
            topic=["Remuneration policy", "Pay and performance alignment"],
            quantity="Key Actions",
            search_terms=[
                "remuneration policy", "compensation", "incentive plan",
                "long-term incentive", "remuneration committee report",
            ],
            knowledge=(
                "Explains how directors’ and executives’ pay is determined, the role of the "
                "remuneration committee, and links between pay, ESG and long‑term results."
            ),
        ),
        ESGMetadataRecord(
            aspect="G10. Shareholder Engagement",
            kpi="Description of policies and channels for communicating with and engaging shareholders.",
            topic=["Shareholder communication", "Investor engagement"],
            quantity="Key Actions",
            search_terms=[
                "shareholder communication policy", "investor relations",
                "AGM", "general meeting", "minority shareholders",
            ],
            knowledge=(
                "Describes how the company maintains dialogue with shareholders, including meetings, "
                "briefings and feedback mechanisms."
            ),
        ),
        ESGMetadataRecord(
            aspect="G11. Ethics and Whistle‑blowing Governance",
            kpi="Description of the governance framework for ethics, anti‑corruption and whistle‑blowing.",
            topic=["Ethics governance", "Whistle‑blowing governance"],
            quantity="Key Actions",
            search_terms=[
                "code of ethics", "whistleblowing policy", "ethics committee",
                "disciplinary procedures",
            ],
            knowledge=(
                "Covers the high‑level governance of ethical conduct, including codes, committees, "
                "whistle‑blowing arrangements and oversight responsibilities."
            ),
        ),
    ],
}


# ----------------------------------------------------------------------
# ESGMetadataModule: convenient accessors for RAG + LLM Agent
# ----------------------------------------------------------------------


class ESGMetadataModule:
    """
    Helper around ESG_METADATA for:
      - listing aspects
      - filtering by aspect
      - searching by topic / keyword
      - exporting to JSON for inspection or downstream use
    """

    def __init__(
        self,
        metadata: Dict[str, List[ESGMetadataRecord]] | None = None,
        auto_validate: bool = False,
    ):
        self._metadata: Dict[str, List[ESGMetadataRecord]] = metadata or ESG_METADATA
        self.schema_version = METADATA_SCHEMA_VERSION
        self.schema_changelog = METADATA_CHANGELOG
        self._validation_errors: List[str] = []
        if auto_validate:
            self._validation_errors = validate_metadata(self._metadata)

    # ---------- basic iteration ----------

    def _all_records(self) -> Iterable[ESGMetadataRecord]:
        for group in self._metadata.values():
            for rec in group:
                yield rec

    # ---------- schema helpers ----------

    def get_schema_info(self) -> Dict[str, Any]:
        return {
            "version": self.schema_version,
            "changelog": self.schema_changelog,
            "validation_errors": self._validation_errors,
        }

    def validate(self, raise_on_error: bool = False) -> List[str]:
        errors = validate_metadata(self._metadata)
        self._validation_errors = errors
        if raise_on_error and errors:
            raise ValueError("Metadata validation failed", errors)
        return errors

    # ---------- public API ----------

    def list_aspects(self) -> List[str]:
        """Return sorted unique list of all Aspect names across pillars."""
        aspects = {rec.aspect for rec in self._all_records()}
        return sorted(aspects)

    def list_by_aspect(self, aspect: str) -> List[ESGMetadataRecord]:
        """
        Get all metadata records belonging to a given Aspect
        (e.g. 'A1. Emissions', 'B2. Health and Safety', 'G7. Risk Management and Internal Control').
        """
        aspect_lower = aspect.lower()
        return [rec for rec in self._all_records() if rec.aspect.lower() == aspect_lower]

    def search_by_topic(self, keyword: str) -> List[ESGMetadataRecord]:
        """
        Full‑text search in Topic, KPI and search_terms.
        Case‑insensitive simple substring matching.
        """
        k = keyword.lower()
        matches: List[ESGMetadataRecord] = []
        for rec in self._all_records():
            in_topic = any(k in t.lower() for t in rec.topic)
            in_terms = any(k in t.lower() for t in rec.search_terms)
            in_synonyms = any(k in s.lower() for s in rec.multilingual_synonyms)
            in_kpi = k in rec.kpi.lower()
            if in_topic or in_terms or in_synonyms or in_kpi:
                matches.append(rec)
        return matches

    def to_json(self, group_by: str = "pillar") -> str:
        """
        Export metadata as JSON string.

        group_by:
          - 'pillar'  -> dict[pillar] -> list[records]
          - 'aspect'  -> dict[aspect] -> list[records]
          - 'flat'    -> list[records]
        """
        if group_by == "pillar":
            data = {
                pillar: [rec.to_json() for rec in records]
                for pillar, records in self._metadata.items()
            }
        elif group_by == "aspect":
            by_aspect: Dict[str, List[Dict]] = {}
            for rec in self._all_records():
                by_aspect.setdefault(rec.aspect, []).append(rec.to_json())
            data = by_aspect
        else:  # flat
            data = [rec.to_json() for rec in self._all_records()]

        return json.dumps(data, ensure_ascii=False, indent=2)

    def save_json(self, path: str | Path, group_by: str = "pillar") -> Path:
        """Persist metadata as JSON on disk and return the destination path."""
        destination = Path(path)
        destination.write_text(self.to_json(group_by=group_by), encoding="utf-8")
        return destination

    def build_prompt_from_metadata(
        self, record: ESGMetadataRecord, reference_text: str
    ) -> str:
        """
        Build an ESGReveal-style prompt for LLM extraction given:
          - An ESG metadata record (Aspect, KPI, Topic, Quantity, etc.)
          - Retrieved reference text from the knowledge base
        
        Returns a formatted prompt string for the LLM.
        """
        topics_str = ", ".join(record.topic)
        optional_lines = []
        if record.canonical_units:
            optional_lines.append(f"Expected Units: {', '.join(record.canonical_units)}")
        if record.dependency_groups:
            optional_lines.append(f"Dependencies: {', '.join(record.dependency_groups)}")
        optional_block = "\n".join(optional_lines)

        prompt = f"""You are an ESG data extraction agent. Your task is to extract information from the reference content below.

**ESG Indicator:**
- Aspect: {record.aspect}
- KPI: {record.kpi}
- Topics: {topics_str}
- Quantity Type: {record.quantity}
    - Indicator ID: {record.indicator_id}
    - Evidence Type: {record.evidence_type}
    - Priority: {record.priority}
    {optional_block}

**Background Knowledge:**
{record.knowledge}

**Reference Content:**
{reference_text}

**Instructions:**
Based strictly on the reference content above, extract the relevant ESG information for this indicator.
Respond in JSON format with the following fields:
- "Disclosure": Whether the indicator is disclosed (Yes/No/Partial)
- "KPI": The KPI being measured
- "Topic": The specific topic addressed
- "Value": Numerical or textual value found (or "Not Available")
- "Unit": Unit of measurement (if applicable, or "N/A")
- "Target": Any targets mentioned (or "Not Available")
- "Action": Actions or measures described (or "Not Available")

If information is not available in the reference content, explicitly state "Not Available" for that field.
"""
        return prompt


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    module = ESGMetadataModule()

    print("All aspects:")
    print(module.list_aspects())

    print("\nRecords for A1. Emissions:")
    for r in module.list_by_aspect("A1. Emissions"):
        print(" -", r.kpi)

    print("\nSearch 'gender':")
    for r in module.search_by_topic("gender"):
        print(f"[{r.aspect}] {r.kpi}")

    print("\nJSON grouped by aspect (truncated):")
    print(module.to_json(group_by="aspect")[:500], "...")

    output_path = Path("esg_metadata.json")
    module.save_json(output_path)
    print(f"\nFull metadata exported to {output_path.resolve()}")
