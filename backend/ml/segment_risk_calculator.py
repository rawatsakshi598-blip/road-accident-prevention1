# backend/ml/segment_risk_calculator.py

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from config import (
    DIGITAL_TWIN_DIR,
    HISTORICAL_RISK_WEIGHT,
    PREDICTIVE_RISK_WEIGHT,
    RISK_CATEGORIES,
    RISK_COLORS,
    SEVERITY_CODE_MAP,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentRiskCalculator:
    """
    Calculate risk scores for road segments based on:
    1. Historical accident data
    2. ML model predictions (optional)
    """

    def __init__(self, segment_mapping: dict, city_key: str = "delhi",
                 predictor=None):
        """
        Initialize risk calculator.

        Args:
            segment_mapping: Dict from AccidentSegmentMapper
            city_key: City identifier
            predictor: Optional AccidentPredictor instance
                       for ML-based risk (can be None)
        """
        self.segment_mapping = segment_mapping
        self.city_key = city_key
        self.predictor = predictor

        # Output paths
        self.output_dir = os.path.join(DIGITAL_TWIN_DIR, city_key)
        os.makedirs(self.output_dir, exist_ok=True)

        self.risks_path = os.path.join(
            self.output_dir, "segment_risks.json"
        )
        self.risk_stats_path = os.path.join(
            self.output_dir, "risk_stats.json"
        )

        # Results storage
        self.segment_risks = {}
        self.risk_stats = {}

        logger.info(
            f"SegmentRiskCalculator initialized for {city_key}: "
            f"{len(segment_mapping):,} segments"
        )

    # ─────────────────────────────────────────
    # HISTORICAL RISK CALCULATION
    # ─────────────────────────────────────────

    def calculate_historical_risk(self, segment_data: dict) -> float:
        """
        Calculate risk score based on accident history from REAL Delhi data.

        Formula:
        - Accident density (accidents per km per year)
        - Weighted by severity (Fatal=10, Grievous=5, Minor=2, NoInjury=1)
        - Fatality rate multiplier
        - Year spread bonus (more years = more established pattern)
        - Normalized to 0-100 scale

        Args:
            segment_data: Segment data dict with accidents

        Returns:
            Historical risk score (0-100)
        """
        total_accidents = segment_data.get("total_accidents", 0)

        if total_accidents == 0:
            return 0.0

        # Segment length in km
        length_m = segment_data.get("length_m", 100)
        length_km = max(length_m / 1000, 0.01)

        # Accident counts by severity
        sev_dist = segment_data.get("severity_distribution", {})
        fatal = sev_dist.get("Fatal", 0)
        grievous = sev_dist.get("Grievous", 0)
        minor = sev_dist.get("Minor", 0)
        no_injury = sev_dist.get("No Injury", 0)

        # Weighted severity score (higher weights for serious outcomes)
        severity_weights = {
            "Fatal": 10.0,
            "Grievous": 5.0,
            "Minor": 2.0,
            "No Injury": 1.0,
        }

        weighted_accidents = (
            fatal * severity_weights["Fatal"] +
            grievous * severity_weights["Grievous"] +
            minor * severity_weights["Minor"] +
            no_injury * severity_weights["No Injury"]
        )

        # Determine years of data from year_distribution if available
        year_dist = segment_data.get("year_distribution", {})
        if year_dist and any(k != "unknown" and k != "" for k in year_dist.keys()):
            distinct_years = len([k for k in year_dist.keys() if k and k != "unknown"])
            years_of_data = max(distinct_years, 1)
        else:
            years_of_data = 5.0  # Delhi data spans 2016-2024

        # Accident density per km per year
        density = weighted_accidents / (length_km * years_of_data)

        # Fatality rate multiplier (higher fatality = higher risk)
        fatal_rate = segment_data.get("fatal_rate", 0)
        fatality_multiplier = 1.0 + fatal_rate * 2.0  # Up to 3x for 100% fatality

        # Apply multiplier
        adjusted_density = density * fatality_multiplier

        # Normalize to 0-100 scale (balanced: /25 gives good spread)
        risk_score = min(100, (adjusted_density / 25) * 100)

        # Small fatal boost only
        if fatal > 0:
            fatal_boost = min(15, fatal * 3)
            risk_score = min(100, risk_score + fatal_boost)

        return round(risk_score, 2)

    # ─────────────────────────────────────────
    # PREDICTIVE RISK CALCULATION
    # ─────────────────────────────────────────

    def calculate_predictive_risk(self, segment_data: dict) -> float:
        """
        Calculate risk using ML model predictions.

        Uses the trained accident severity predictor
        to predict risk for "typical" accident conditions
        on this segment.

        Args:
            segment_data: Segment data dict

        Returns:
            Predictive risk score (0-100)
            Returns 50 (moderate) if no predictor available
        """
        if self.predictor is None:
            # No ML model available, return 0 (let historical dominate)
            return 0.0

        try:
            # Create a "typical" accident scenario for this segment
            # Based on most common conditions in segment history

            # Most common weather
            weather_dist = segment_data.get("weather_distribution", {})
            most_common_weather = max(
                weather_dist, key=weather_dist.get
            ) if weather_dist else "Clear"

            # Most common time period
            time_dist = segment_data.get("time_distribution", {})
            most_common_time = max(
                time_dist, key=time_dist.get
            ) if time_dist else "Afternoon"

            # Road type
            road_type = segment_data.get("road_type", "unknown")

            # Create dummy input for prediction
            # This would match your actual prediction input schema
            dummy_input = {
                "Day_of_Week": "Monday",
                "Time_of_Accident": "12:00",
                "Accident_Location": segment_data.get("road_name", "Unknown"),
                "Nature_of_Accident": "Unknown",
                "Causes_D1": "Other",
                "Road_Condition": "Dry",
                "Weather_Conditions": most_common_weather,
                "Intersection_Type": "Unknown",
                "Vehicle_Type_V1": "Car/Jeep/Van",
                "Vehicle_Type_V2": "",
                "Number_of_Vehicles": 2,
            }

            # Get prediction
            result = self.predictor.predict(dummy_input)

            # Extract severity probabilities
            probabilities = result.get("probabilities", {})

            # Calculate weighted risk from probability distribution
            weighted_risk = (
                probabilities.get("Fatal", 0) * 100 +
                probabilities.get("Grievous", 0) * 75 +
                probabilities.get("Minor", 0) * 50 +
                probabilities.get("No Injury", 0) * 25
            )

            return round(weighted_risk, 2)

        except Exception as e:
            logger.warning(f"Predictive risk calculation failed: {e}")
            return 0.0

    # ─────────────────────────────────────────
    # COMPOSITE RISK CALCULATION
    # ─────────────────────────────────────────

    def calculate_composite_risk(self, segment_data: dict) -> dict:
        """
        Calculate final composite risk score.

        Combines historical and predictive risk using weights.

        Args:
            segment_data: Segment data dict

        Returns:
            Dict with all risk components
        """
        historical_risk = self.calculate_historical_risk(segment_data)
        predictive_risk = self.calculate_predictive_risk(segment_data)

        # Composite risk: use max so high historical risk isn't dragged down
        composite_risk = max(historical_risk, predictive_risk)
        # If both contribute, give a small bonus
        if historical_risk > 0 and predictive_risk > 0:
            composite_risk = min(100, composite_risk * 1.1)

        composite_risk = round(composite_risk, 2)

        # Determine risk category
        risk_category = "Moderate"
        for category, (low, high) in RISK_CATEGORIES.items():
            if low <= composite_risk < high:
                risk_category = category
                break

        # Get risk color
        risk_color = RISK_COLORS.get(risk_category, "#ffff00")

        return {
            "historical_risk": historical_risk,
            "predictive_risk": predictive_risk,
            "composite_risk": composite_risk,
            "risk_category": risk_category,
            "risk_color": risk_color,
        }

    # ─────────────────────────────────────────
    # BATCH CALCULATION
    # ─────────────────────────────────────────

    def calculate_all_segments(self) -> dict:
        """
        Calculate risk scores for all segments using rank-based normalization.
        This guarantees visible green/blue/yellow/orange/red buckets even when
        raw accident counts are heavily skewed.
        """
        logger.info("=" * 50)
        logger.info("Calculating segment risk scores...")
        logger.info("=" * 50)

        records = []

        for idx, (segment_id, segment_data) in enumerate(self.segment_mapping.items(), 1):
            if idx % 100 == 0:
                logger.info(f"Processed {idx:,} segments...")

            try:
                sev_dist = segment_data.get("severity_distribution", {})
                fatal = int(sev_dist.get("Fatal", 0) or 0)
                grievous = int(sev_dist.get("Grievous", 0) or 0)
                minor = int(sev_dist.get("Minor", 0) or 0)
                no_injury = int(sev_dist.get("No Injury", 0) or 0)
                total_accidents = int(segment_data.get("total_accidents", 0) or 0)

                # Raw importance score; only used for ranking
                raw_score = (
                    fatal * 12.0 +
                    grievous * 6.0 +
                    minor * 2.0 +
                    no_injury * 1.0 +
                    total_accidents * 0.25
                )

                records.append({
                    "segment_id": segment_id,
                    "segment_data": segment_data,
                    "raw_score": raw_score,
                })

            except Exception as e:
                logger.warning(f"Score collection failed for {segment_id}: {e}")
                continue

        if not records:
            self.segment_risks = {}
            self.risk_stats = {}
            return {}

        # Sort ascending by raw score for percentile/rank mapping
        records.sort(key=lambda x: x["raw_score"])
        n = len(records)

        def rank_to_risk(rank: int) -> float:
            """
            Map rank percentile to 0-100 risk with guaranteed buckets:
            0-10   => green
            10-40  => blue
            40-60  => yellow
            60-80  => orange
            80-100 => red
            """
            p = rank / max(n - 1, 1)

            if p < 0.10:
                # Green bucket: 5-9.9
                t = p / 0.10
                return 5.0 + t * 4.9
            elif p < 0.40:
                # Blue bucket: 10-39.9
                t = (p - 0.10) / 0.30
                return 10.0 + t * 29.9
            elif p < 0.60:
                # Yellow bucket: 40-59.9
                t = (p - 0.40) / 0.20
                return 40.0 + t * 19.9
            elif p < 0.80:
                # Orange bucket: 60-79.9
                t = (p - 0.60) / 0.20
                return 60.0 + t * 19.9
            else:
                # Red bucket: 80-100
                t = (p - 0.80) / 0.20
                return 80.0 + t * 20.0

        risk_distribution = {
            "No Risk": 0,
            "Low": 0,
            "Moderate": 0,
            "High": 0,
            "Very High": 0,
        }

        results = {}

        for rank, rec in enumerate(records):
            segment_id = rec["segment_id"]
            segment_data = rec["segment_data"]
            composite_risk = round(rank_to_risk(rank), 2)

            risk_category = "No Risk"
            for category, (low, high) in RISK_CATEGORIES.items():
                if low <= composite_risk < high:
                    risk_category = category
                    break

            risk_color = RISK_COLORS.get(risk_category, "#22C55E")

            results[segment_id] = {
                "segment_id": segment_id,
                "road_name": segment_data.get("road_name", "Unknown"),
                "road_type": segment_data.get("road_type", "unknown"),
                "length_m": segment_data.get("length_m", 0),
                "centroid_lat": segment_data.get("centroid_lat", 0),
                "centroid_lon": segment_data.get("centroid_lon", 0),
                "total_accidents": int(segment_data.get("total_accidents", 0) or 0),
                "fatal_count": int(segment_data.get("severity_distribution", {}).get("Fatal", 0) or 0),
                "grievous_count": int(segment_data.get("severity_distribution", {}).get("Grievous", 0) or 0),
                "minor_count": int(segment_data.get("severity_distribution", {}).get("Minor", 0) or 0),
                "historical_risk": composite_risk,
                "predictive_risk": 0.0,
                "composite_risk": composite_risk,
                "risk_score": composite_risk,
                "risk_category": risk_category,
                "risk_color": risk_color,
                "color": risk_color,
                "is_virtual": bool(segment_data.get("is_virtual", False)),
                "raw_score": round(rec["raw_score"], 2),
            }

            risk_distribution[risk_category] += 1

        risk_scores_list = [v["composite_risk"] for v in results.values()]
        self.risk_stats = {
            "total_segments": len(results),
            "risk_distribution": risk_distribution,
            "mean_risk": round(np.mean(risk_scores_list), 2) if risk_scores_list else 0,
            "median_risk": round(np.median(risk_scores_list), 2) if risk_scores_list else 0,
            "std_risk": round(np.std(risk_scores_list), 2) if risk_scores_list else 0,
            "min_risk": round(min(risk_scores_list), 2) if risk_scores_list else 0,
            "max_risk": round(max(risk_scores_list), 2) if risk_scores_list else 0,
            "high_risk_segments": risk_distribution["High"] + risk_distribution["Very High"],
            "calculated_at": datetime.now().isoformat(),
        }

        logger.info(f"\nRisk calculation complete: {len(results):,} segments")
        logger.info(f"Risk distribution: {risk_distribution}")
        logger.info(f"High/Very High risk segments: {self.risk_stats['high_risk_segments']:,}")

        self.segment_risks = results
        return results

    def get_segment_risk(self, segment_id: str) -> dict:
        """
        Get risk data for a specific segment.

        Args:
            segment_id: Segment identifier

        Returns:
            Risk data dict or None if not found
        """
        return self.segment_risks.get(segment_id, None)

    def get_top_dangerous_segments(self, n: int = 10,
                                    min_risk: float = 0) -> list:
        """
        Get N most dangerous segments.

        Args:
            n: Number of segments to return
            min_risk: Minimum risk threshold

        Returns:
            List of segment risk dicts, sorted by risk (descending)
        """
        # Filter by minimum risk
        filtered = [
            seg for seg in self.segment_risks.values()
            if seg["composite_risk"] >= min_risk
        ]

        # Sort by composite risk
        sorted_segments = sorted(
            filtered,
            key=lambda x: x["composite_risk"],
            reverse=True
        )

        return sorted_segments[:n]

    def get_segments_by_category(self, category: str) -> list:
        """
        Get all segments in a specific risk category.

        Args:
            category: Risk category name
                      ("Very Low", "Low", "Moderate", "High", "Very High")

        Returns:
            List of segment risk dicts
        """
        return [
            seg for seg in self.segment_risks.values()
            if seg["risk_category"] == category
        ]

    # ─────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────

    def save_risks(self, risks: dict = None):
        """
        Save segment risks to JSON file.

        Args:
            risks: Risk dict to save (uses self.segment_risks if None)
        """
        if risks is None:
            risks = self.segment_risks

        with open(self.risks_path, "w") as f:
            json.dump(risks, f, indent=2)

        with open(self.risk_stats_path, "w") as f:
            json.dump(self.risk_stats, f, indent=2)

        size_kb = os.path.getsize(self.risks_path) / 1024
        logger.info(
            f"Risks saved: {self.risks_path} ({size_kb:.1f} KB)"
        )
        logger.info(f"Stats saved: {self.risk_stats_path}")

    def load_risks(self) -> dict:
        """
        Load segment risks from JSON file.

        Returns:
            Risk dict or empty dict if not found
        """
        if not os.path.exists(self.risks_path):
            logger.warning(f"No risks found at {self.risks_path}")
            return {}

        with open(self.risks_path, "r") as f:
            risks = json.load(f)

        if os.path.exists(self.risk_stats_path):
            with open(self.risk_stats_path, "r") as f:
                self.risk_stats = json.load(f)

        logger.info(f"Risks loaded: {len(risks):,} segments")
        self.segment_risks = risks
        return risks

    def is_risks_valid(self) -> bool:
        """Check if saved risks exist."""
        return os.path.exists(self.risks_path)

    def get_stats(self) -> dict:
        """Get risk statistics."""
        if not self.risk_stats and os.path.exists(self.risk_stats_path):
            with open(self.risk_stats_path, "r") as f:
                self.risk_stats = json.load(f)
        return self.risk_stats
