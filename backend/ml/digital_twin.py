# backend/ml/digital_twin.py

import os
import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd

from config import (
    DIGITAL_TWIN_DIR,
    CITIES_CONFIG,
    RISK_CATEGORIES,
    RISK_COLORS,
)
from ml.road_network_loader import RoadNetworkLoader
from ml.delhi_data_mapper import DelhiDataMapper
from ml.segment_risk_calculator import SegmentRiskCalculator
from ml.scenario_simulator import ScenarioSimulator
from ml.heatmap_generator import HeatmapGenerator

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitalTwin:
    """
    Digital Twin of Road Network.

    Main orchestrator class that coordinates all components:
    1. Road network loading (OSMnx)
    2. Accident-to-segment mapping (using REAL Delhi datasets via DelhiDataMapper)
    3. Risk calculation
    4. Scenario simulation
    5. Heatmap generation

    Provides caching for fast reloading.
    """

    def __init__(self, city_key: str = "delhi", predictor=None):
        """
        Initialize Digital Twin for a city.

        Args:
            city_key: City identifier from CITIES_CONFIG
            predictor: Optional AccidentPredictor for ML-based risk
        """
        if city_key not in CITIES_CONFIG:
            raise ValueError(
                f"City '{city_key}' not found in CITIES_CONFIG. "
                f"Available: {list(CITIES_CONFIG.keys())}"
            )

        self.city_key = city_key
        self.city_config = CITIES_CONFIG[city_key]
        self.city_name = self.city_config["name"]
        self.predictor = predictor

        # Component instances (initialized lazily)
        self.network_loader: Optional[RoadNetworkLoader] = None
        self.data_mapper: Optional[DelhiDataMapper] = None
        self.risk_calculator: Optional[SegmentRiskCalculator] = None
        self.scenario_simulator: Optional[ScenarioSimulator] = None
        self.heatmap_generator: Optional[HeatmapGenerator] = None

        # Data storage
        self.road_network = None
        self.edges_gdf = None
        self.segment_mapping = {}
        self.segment_risks = {}

        # Metadata
        self.metadata = {
            "city_key": city_key,
            "city_name": self.city_name,
            "built_at": None,
            "status": "not_initialized",
        }

        # Output paths
        self.output_dir = os.path.join(DIGITAL_TWIN_DIR, city_key)
        os.makedirs(self.output_dir, exist_ok=True)

        self.metadata_path = os.path.join(
            self.output_dir, "twin_metadata.json"
        )
        self.top_dangerous_path = os.path.join(
            self.output_dir, "top_dangerous.json"
        )

        logger.info(f"DigitalTwin initialized for: {self.city_name}")

    # ─────────────────────────────────────────
    # BUILD PIPELINE
    # ─────────────────────────────────────────

    def build_twin(self, force_rebuild: bool = False) -> dict:
        """
        Build complete digital twin from scratch.

        Pipeline:
        1. Load road network (OSMnx)
        2. Map REAL Delhi accidents to segments (via DelhiDataMapper)
        3. Calculate risk scores
        4. Generate heatmaps

        Args:
            force_rebuild: Force rebuild even if cache exists

        Returns:
            Build summary dict
        """
        logger.info("=" * 60)
        logger.info(f"BUILDING DIGITAL TWIN: {self.city_name}")
        logger.info(f"Using REAL Delhi Police Data via DelhiDataMapper")
        if force_rebuild:
            logger.info("FORCE REBUILD: All caches will be ignored")
        else:
            logger.info("CACHE-AWARE: Will use cached data where available")
            logger.info("(Use force_rebuild=True to recompute everything from scratch)")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Load road network
            logger.info("\n[1/4] Loading road network...")
            step1_start = time.time()
            self._build_road_network(force_rebuild)
            step1_time = time.time() - step1_start
            logger.info(f"Step 1 complete ({step1_time:.1f}s)")

            # Step 2: Map REAL Delhi accidents
            logger.info("\n[2/4] Mapping REAL Delhi accidents to segments...")
            step2_start = time.time()
            self._build_accident_mapping(force_rebuild)
            step2_time = time.time() - step2_start
            logger.info(f"Step 2 complete ({step2_time:.1f}s)")

            # Step 3: Calculate risks
            logger.info("\n[3/4] Calculating segment risk scores...")
            step3_start = time.time()
            self._build_risk_scores(force_rebuild)
            step3_time = time.time() - step3_start
            logger.info(f"Step 3 complete ({step3_time:.1f}s)")

            # Step 4: Generate heatmaps
            logger.info("\n[4/4] Generating heatmaps...")
            step4_start = time.time()
            self._build_heatmaps(force_rebuild)
            step4_time = time.time() - step4_start
            logger.info(f"Step 4 complete ({step4_time:.1f}s)")

            # Initialize simulator
            self._initialize_simulator()

            total_time = time.time() - start_time

            # Update metadata
            self.metadata.update({
                "built_at": datetime.now().isoformat(),
                "build_duration_seconds": round(total_time, 2),
                "status": "ready",
                "total_segments": len(self.edges_gdf),
                "mapped_accidents": sum(
                    v.get("total_accidents", 0)
                    for v in self.segment_risks.values()
                ),
                "segments_with_accidents": len(self.segment_mapping),
                "high_risk_segments": len([
                    v for v in self.segment_risks.values()
                    if v.get("composite_risk", 0) >= 60
                ]),
                "data_source": "DelhiDataMapper (REAL Delhi Police Data)",
                "step_times": {
                    "road_network": round(step1_time, 2),
                    "accident_mapping": round(step2_time, 2),
                    "risk_calculation": round(step3_time, 2),
                    "heatmap_generation": round(step4_time, 2),
                },
            })

            self._save_metadata()
            self._generate_top_dangerous()

            logger.info("\n" + "=" * 60)
            logger.info("DIGITAL TWIN BUILD COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            logger.info(f"Total segments: {self.metadata['total_segments']:,}")
            logger.info(f"Mapped accidents: {self.metadata['mapped_accidents']:,}")
            logger.info(f"High-risk segments: {self.metadata['high_risk_segments']:,}")
            logger.info("=" * 60)

            return self.metadata

        except Exception as e:
            logger.error(f"Build failed: {e}")
            self.metadata["status"] = "build_failed"
            self.metadata["error"] = str(e)
            self._save_metadata()
            raise

    def _build_road_network(self, force_rebuild: bool):
        """Step 1: Load road network."""
        self.network_loader = RoadNetworkLoader(self.city_key)

        if force_rebuild or not self.network_loader.is_cache_valid():
            if force_rebuild and self.network_loader.is_cache_valid():
                logger.info("  Force rebuild: Ignoring cached road network")
            logger.info("  Downloading road network from OpenStreetMap (1-5 min)...")
            self.road_network = self.network_loader.get_or_download_network()
        else:
            logger.info("  Using cached road network (loaded in seconds)")
            self.road_network = self.network_loader.load_cached_network()

        self.edges_gdf = self.network_loader.get_edges_gdf()

    def _build_accident_mapping(self, force_rebuild: bool):
        """Step 2: Map REAL Delhi accidents to segments using DelhiDataMapper."""
        self.data_mapper = DelhiDataMapper(
            self.edges_gdf, self.city_key
        )

        if force_rebuild or not self.data_mapper.is_mapping_valid():
            if force_rebuild and self.data_mapper.is_mapping_valid():
                logger.info("  Force rebuild: Ignoring cached accident mapping")
            logger.info("  Mapping accidents to road segments (2-5 min)...")
            self.segment_mapping = self.data_mapper.geocode_and_map_all()
            self.data_mapper.save_mapping(self.segment_mapping)
        else:
            logger.info("  Using cached accident mapping (loaded in seconds)")
            self.segment_mapping = self.data_mapper.load_mapping()

        # Log virtual vs real segment counts
        virtual_count = sum(
            1 for v in self.segment_mapping.values()
            if v.get("is_virtual", False)
        )
        real_count = len(self.segment_mapping) - virtual_count

        logger.info(
            f"Accident mapping: {len(self.segment_mapping)} segments with accidents "
            f"({real_count} real, {virtual_count} virtual)"
        )

    def _build_risk_scores(self, force_rebuild: bool):
        """Step 3: Calculate risk scores."""
        self.risk_calculator = SegmentRiskCalculator(
            self.segment_mapping,
            self.city_key,
            self.predictor
        )

        if force_rebuild or not self.risk_calculator.is_risks_valid():
            if force_rebuild and self.risk_calculator.is_risks_valid():
                logger.info("  Force rebuild: Ignoring cached risk scores")
            logger.info("  Calculating risk scores from scratch...")
            self.segment_risks = self.risk_calculator.calculate_all_segments()
            self.risk_calculator.save_risks(self.segment_risks)# backend/ml/digital_twin.py

        if force_rebuild or not self.network_loader.is_cache_valid():
            if force_rebuild and self.network_loader.is_cache_valid():
                logger.info("  Force rebuild: Ignoring cached road network")
            logger.info("  Downloading road network from OpenStreetMap (1-5 min)...")
            self.road_network = self.network_loader.get_or_download_network()
        else:
            logger.info("  Using cached road network (loaded in seconds)")
            self.road_network = self.network_loader.load_cached_network()

        self.edges_gdf = self.network_loader.get_edges_gdf()

    def _build_accident_mapping(self, force_rebuild: bool):
        """Step 2: Map REAL Delhi accidents to segments using DelhiDataMapper."""
        self.data_mapper = DelhiDataMapper(
            self.edges_gdf, self.city_key
        )

        if force_rebuild or not self.data_mapper.is_mapping_valid():
            if force_rebuild and self.data_mapper.is_mapping_valid():
                logger.info("  Force rebuild: Ignoring cached accident mapping")
            logger.info("  Mapping accidents to road segments (2-5 min)...")
            self.segment_mapping = self.data_mapper.geocode_and_map_all()
            self.data_mapper.save_mapping(self.segment_mapping)
        else:
            logger.info("  Using cached accident mapping (loaded in seconds)")
            self.segment_mapping = self.data_mapper.load_mapping()

        # Log virtual vs real segment counts
        virtual_count = sum(
            1 for v in self.segment_mapping.values()
            if v.get("is_virtual", False)
        )
        real_count = len(self.segment_mapping) - virtual_count

        logger.info(
            f"Accident mapping: {len(self.segment_mapping)} segments with accidents "
            f"({real_count} real, {virtual_count} virtual)"
        )

    def _build_risk_scores(self, force_rebuild: bool):
        """Step 3: Calculate risk scores."""
        self.risk_calculator = SegmentRiskCalculator(
            self.segment_mapping,
            self.city_key,
            self.predictor
        )

        if force_rebuild or not self.risk_calculator.is_risks_valid():
            if force_rebuild and self.risk_calculator.is_risks_valid():
                logger.info("  Force rebuild: Ignoring cached risk scores")
            logger.info("  Calculating risk scores from scratch...")
            self.segment_risks = self.risk_calculator.calculate_all_segments()
            self.risk_calculator.save_risks(self.segment_risks)
        else:
            logger.info("  Using cached risk scores (loaded in seconds)")
            self.segment_risks = self.risk_calculator.load_risks()

        # Enrich segment_risks with is_virtual and minor_count from segment_mapping
        for seg_id, risk_data in self.segment_risks.items():
            mapping_data = self.segment_mapping.get(seg_id, {})

            # Add is_virtual flag (from DelhiDataMapper)
            risk_data["is_virtual"] = mapping_data.get("is_virtual", False)

            # Add minor_count for API response
            sev_dist = mapping_data.get("severity_distribution", {})
            risk_data["minor_count"] = sev_dist.get("Minor", 0)

    def _build_heatmaps(self, force_rebuild: bool):
        """Step 4: Generate heatmaps."""
        self.heatmap_generator = HeatmapGenerator(
            self.edges_gdf,
            self.segment_risks,
            self.city_key
        )

        if force_rebuild or not self.heatmap_generator.is_heatmaps_valid():
            print(json.dumps(metadata, indent=2))
        else:
            logger.info("  Using cached risk scores (loaded in seconds)")
            self.segment_risks = self.risk_calculator.load_risks()

        # Enrich segment_risks with is_virtual and minor_count from segment_mapping
        for seg_id, risk_data in self.segment_risks.items():
            mapping_data = self.segment_mapping.get(seg_id, {})

            # Add is_virtual flag (from DelhiDataMapper)
            risk_data["is_virtual"] = mapping_data.get("is_virtual", False)

            # Add minor_count for API response
            sev_dist = mapping_data.get("severity_distribution", {})
            risk_data["minor_count"] = sev_dist.get("Minor", 0)

    def _build_heatmaps(self, force_rebuild: bool):
        """Step 4: Generate heatmaps."""
        self.heatmap_generator = HeatmapGenerator(
            self.edges_gdf,
            self.segment_risks,
            self.city_key
        )

        if force_rebuild or not self.heatmap_generator.is_heatmaps_valid():
            if force_rebuild and self.heatmap_generator.is_heatmaps_valid():
                logger.info("  Force rebuild: Ignoring cached heatmaps")
            logger.info("  Generating heatmaps from scratch...")
            self.heatmap_generator.save_heatmaps()
        else:
            logger.info("  Using cached heatmaps (loaded in seconds)")

    def _initialize_simulator(self):
        """Initialize scenario simulator."""
        self.scenario_simulator = ScenarioSimulator(
            self.predictor,
            self.segment_risks,
            self.city_key
        )
        if self.predictor is not None:
            logger.info("Scenario simulator initialized with ML predictor")
        else:
            logger.info("Scenario simulator initialized with rule-based scoring")

    # ─────────────────────────────────────────
    # LOAD FROM CACHE
    # ─────────────────────────────────────────

    def load_twin(self) -> dict:
        """
        Load digital twin from cache.

        Returns:
            Metadata dict
        """
        logger.info(f"Loading digital twin from cache: {self.city_name}")

        start_time = time.time()

        try:
            # Load network
            self.network_loader = RoadNetworkLoader(self.city_key)
            if not self.network_loader.is_cache_valid():
                raise FileNotFoundError(
                    "Road network cache not found. Run build_twin() first."
                )
            self.road_network = self.network_loader.load_cached_network()
            self.edges_gdf = self.network_loader.get_edges_gdf()

            # Load accident mapping (using DelhiDataMapper)
            self.data_mapper = DelhiDataMapper(
                self.edges_gdf, self.city_key
            )
            if not self.data_mapper.is_mapping_valid():
                raise FileNotFoundError(
                    "Accident mapping cache not found. Run build_twin() first."
                )
            self.segment_mapping = self.data_mapper.load_mapping()

            # Load risks
            self.risk_calculator = SegmentRiskCalculator(
                self.segment_mapping, self.city_key, self.predictor
            )
            if not self.risk_calculator.is_risks_valid():
                raise FileNotFoundError(
                    "Risk scores cache not found. Run build_twin() first."
                )
            self.segment_risks = self.risk_calculator.load_risks()

            # Enrich segment_risks with is_virtual and minor_count from segment_mapping
            for seg_id, risk_data in self.segment_risks.items():
                mapping_data = self.segment_mapping.get(seg_id, {})
                risk_data["is_virtual"] = mapping_data.get("is_virtual", False)
                sev_dist = mapping_data.get("severity_distribution", {})
                risk_data["minor_count"] = sev_dist.get("Minor", 0)

            # Load heatmaps
            self.heatmap_generator = HeatmapGenerator(
                self.edges_gdf, self.segment_risks, self.city_key
            )

            # Initialize simulator
            self._initialize_simulator()

            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata["status"] = "loaded_partial"

            load_time = time.time() - start_time

            logger.info(
                f"Digital twin loaded in {load_time:.1f}s: "
                f"{len(self.segment_risks):,} segments"
            )

            return self.metadata

        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise

    # ─────────────────────────────────────────
    # INITIALIZE (AUTO-DETECT CACHE)
    # ─────────────────────────────────────────

    def initialize_twin(self, force_rebuild: bool = False) -> dict:
        """
        Initialize twin: load from cache or build if needed.

        Args:
            force_rebuild: Force rebuild even if cache exists

        Returns:
            Metadata dict
        """
        if force_rebuild:
            return self.build_twin(force_rebuild=True)

        try:
            return self.load_twin()
        except FileNotFoundError:
            logger.info("Cache not found. Building twin from scratch...")
            return self.build_twin()

    # ─────────────────────────────────────────
    # QUERY METHODS
    # ─────────────────────────────────────────

    def get_segment_info(self, segment_id: str) -> Optional[dict]:
        """Get detailed info for a specific segment."""
        if segment_id not in self.segment_risks:
            return None

        risk_data = self.segment_risks[segment_id]
        mapping_data = self.segment_mapping.get(segment_id, {})

        return {
            **risk_data,
            **mapping_data,
        }

    def get_top_dangerous_segments(self, n: int = 10, min_risk: float = 0) -> list:
        """Get N most dangerous segments."""
        if self.risk_calculator:
            return self.risk_calculator.get_top_dangerous_segments(n, min_risk)

        filtered = [
            v for v in self.segment_risks.values()
            if v.get("composite_risk", 0) >= min_risk
        ]
        sorted_segments = sorted(
            filtered,
            key=lambda x: x.get("composite_risk", 0),
            reverse=True
        )
        return sorted_segments[:n]

    # ─────────────────────────────────────────
    # HEATMAP & API RESPONSE
    # ─────────────────────────────────────────

    def get_heatmap_data(self, heatmap_type: str = "grid") -> dict:
        """
        Get heatmap data for visualization.

        Args:
            heatmap_type: 'grid' or 'segments'

        Returns:
            Dict with heatmap data
        """
        if not self.heatmap_generator:
            raise RuntimeError("Twin not initialized")

        if heatmap_type == "grid":
            return self.heatmap_generator.load_grid_heatmap()
        elif heatmap_type == "segments":
            return self.heatmap_generator.load_segment_heatmap()
        else:
            raise ValueError(f"Invalid heatmap_type: {heatmap_type}")

    def get_segments_for_api(self, risk_threshold: float = 0) -> List[dict]:
        """
        Get segment data formatted for the API response.

        Returns segments with all fields required by the frontend:
        - segment_id, road_name, road_type, centroid_lat, centroid_lon
        - total_accidents, fatal_count, grievous_count, minor_count
        - composite_risk (0-100), risk_category, risk_color
        - is_virtual (boolean)

        Args:
            risk_threshold: Minimum composite_risk to include (0 = all)

        Returns:
            List of segment dicts
        """
        segments = []

        for seg_id, risk_data in self.segment_risks.items():
            composite_risk = risk_data.get("composite_risk", 0)

            if composite_risk < risk_threshold:
                continue

            # Determine risk category from composite_risk
            risk_category = "No Risk"
            for category, (low, high) in RISK_CATEGORIES.items():
                if low <= composite_risk < high:
                    risk_category = category
                    break
            if composite_risk >= 80:
                risk_category = "Very High"

            # Get risk color
            risk_color = RISK_COLORS.get(risk_category, "#22C55E")

            segment = {
                "segment_id": risk_data.get("segment_id", seg_id),
                "road_name": risk_data.get("road_name", "Unknown"),
                "road_type": risk_data.get("road_type", "unknown"),
                "centroid_lat": risk_data.get("centroid_lat", 0),
                "centroid_lon": risk_data.get("centroid_lon", 0),
                "total_accidents": risk_data.get("total_accidents", 0),
                "fatal_count": risk_data.get("fatal_count", 0),
                "grievous_count": risk_data.get("grievous_count", 0),
                "minor_count": risk_data.get("minor_count", 0),
                "composite_risk": round(composite_risk, 2),
                "risk_category": risk_data.get("risk_category", risk_category),
                "risk_color": risk_data.get("risk_color", risk_color),
                "is_virtual": risk_data.get("is_virtual", False),
            }

            segments.append(segment)

        # Sort by composite_risk descending
        segments.sort(key=lambda x: x["composite_risk"], reverse=True)

        return segments

    def get_heatmap_api_response(self, heatmap_type: str = "segments",
                                  risk_threshold: float = 0) -> dict:
        """
        Get heatmap data formatted for the /api/digital-twin/heatmap endpoint.

        This method returns a complete response with:
        - City info (key, name, center, zoom, bbox)
        - Color scale legend
        - Segment or grid data depending on type
        - For segment type: includes all required API fields

        Args:
            heatmap_type: 'grid' or 'segments'
            risk_threshold: Minimum risk to include (0 = all)

        Returns:
            Dict formatted for API response
        """
        response = {
            "city_key": self.city_key,
            "city_name": self.city_name,
            "center": self.city_config["center"],
            "zoom_level": self.city_config["zoom_level"],
            "bbox": self.city_config["bbox"],
            "color_scale": self._get_color_scale(),
            "risk_threshold": risk_threshold,
            "generated_at": datetime.now().isoformat(),
        }

        if heatmap_type == "segments":
            # Load full cached segment heatmap so frontend gets coordinates
            segment_heatmap = self.get_heatmap_data("segments")

            if isinstance(segment_heatmap, dict):
                raw_segments = segment_heatmap.get("data", [])
            else:
                raw_segments = []

            segments = []
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue

                # Support both risk_score and composite_risk
                score = seg.get("risk_score", seg.get("composite_risk", 0)) or 0
                if score < risk_threshold:
                    continue

                seg = seg.copy()

                # Ensure frontend-friendly fields exist
                if "risk_score" not in seg or seg.get("risk_score") is None:
                    seg["risk_score"] = round(float(score), 2)

                if "composite_risk" not in seg or seg.get("composite_risk") is None:
                    seg["composite_risk"] = round(float(score), 2)

                if "color" not in seg or not seg.get("color"):
                    seg["color"] = seg.get("risk_color", self._get_color_scale()[0]["color"])

                if "risk_color" not in seg or not seg.get("risk_color"):
                    seg["risk_color"] = seg["color"]

                # coordinates should already be present in heatmap JSON,
                # but keep a safe fallback if missing
                if "coordinates" not in seg or not seg.get("coordinates"):
                    seg["coordinates"] = []

                segments.append(seg)

            response["type"] = "segments"
            response["data"] = segments
            response["segments_count"] = len(segments)

        elif heatmap_type == "grid":
            # Load cached grid heatmap
            grid_data = self.get_heatmap_data("grid")
            response["type"] = "grid"
            response["data"] = grid_data.get("data", [])
            response["grid_size"] = grid_data.get("grid_size", 50)
            response["grid_points_count"] = len(grid_data.get("data", []))

        else:
            raise ValueError(f"Invalid heatmap_type: {heatmap_type}")

        return response

    def _get_color_scale(self) -> list:
        """
        Get color scale for the heatmap legend.

        Returns:
            List of dicts with category, color, range_min, range_max
        """
        scale = []
        for category in ["No Risk", "Low", "Moderate", "High", "Very High"]:
            color = RISK_COLORS[category]
            risk_range = RISK_CATEGORIES[category]
            scale.append({
                "category": category,
                "color": color,
                "range_min": risk_range[0],
                "range_max": risk_range[1] - 1 if risk_range[1] <= 100 else 100,
            })
        return scale

    def simulate_scenario(self, segment_id: str, scenario_type: str, **kwargs) -> dict:
        """Run scenario simulation on a segment."""
        if not self.scenario_simulator:
            self.scenario_simulator = ScenarioSimulator(
                self.predictor,
                self.segment_risks,
                self.city_key
            )

        if scenario_type == "weather":
            return self.scenario_simulator.simulate_weather_change(
                segment_id, kwargs.get("weather", "Clear")
            )
        elif scenario_type == "time":
            return self.scenario_simulator.simulate_time_change(
                segment_id, kwargs.get("time_period", "Day")
            )
        elif scenario_type == "traffic":
            return self.scenario_simulator.simulate_traffic_change(
                segment_id, kwargs.get("traffic_level", "Medium")
            )
        elif scenario_type == "intervention":
            return self.scenario_simulator.simulate_intervention(
                segment_id, kwargs.get("intervention_id", "street_lights")
            )
        else:
            raise ValueError(f"Unknown scenario_type: {scenario_type}")

    # ─────────────────────────────────────────
    # METADATA & STATS
    # ─────────────────────────────────────────

    def get_metadata(self) -> dict:
        """Get digital twin metadata."""
        return self.metadata.copy()

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        stats = {
            "city": self.city_name,
            "city_key": self.city_key,
            "status": self.metadata.get("status", "unknown"),
        }

        if self.risk_calculator:
            stats["risk_stats"] = self.risk_calculator.get_stats()

        if self.data_mapper:
            stats["mapping_stats"] = self.data_mapper.get_stats()

        if self.network_loader:
            stats["network_stats"] = self.network_loader.get_network_stats()

        # Add metadata for frontend
        stats["metadata"] = self.metadata

        # Add color scale for frontend
        stats["color_scale"] = self._get_color_scale()

        return stats

    def _save_metadata(self):
        """Save metadata to JSON."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_top_dangerous(self):
        """Generate and save top dangerous segments list."""
        top = self.get_top_dangerous_segments(n=50)

        with open(self.top_dangerous_path, "w") as f:
            json.dump(top, f, indent=2)

    # ─────────────────────────────────────────
    # REFRESH
    # ─────────────────────────────────────────

    def refresh_twin(self) -> dict:
        """Rebuild twin with latest data."""
        logger.info("Refreshing digital twin...")
        return self.build_twin(force_rebuild=True)


# ─────────────────────────────────────────────
# CLI ENTRY POINT (for testing)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    city_key = sys.argv[1] if len(sys.argv) > 1 else "delhi"
    force_rebuild = "--rebuild" in sys.argv

    print(f"\nBuilding Digital Twin for: {city_key}")
    print(f"Force rebuild: {force_rebuild}\n")

    twin = DigitalTwin(city_key)
    metadata = twin.initialize_twin(force_rebuild=force_rebuild)

    print("\n" + "=" * 60)
    print("DIGITAL TWIN READY")
    print("=" * 60)
    print(json.dumps(metadata, indent=2))