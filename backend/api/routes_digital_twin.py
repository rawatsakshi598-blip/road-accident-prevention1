# backend/api/routes_digital_twin.py

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime
import logging

from config import CITIES_CONFIG
from ml.digital_twin import DigitalTwin

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
router = APIRouter(prefix="/api/twin", tags=["Digital Twin"])

# ─────────────────────────────────────────────
# GLOBAL DIGITAL TWIN REGISTRY
# (Loaded on startup, accessed by routes)
# ─────────────────────────────────────────────
digital_twins = {}


def get_twin(city_key: str) -> DigitalTwin:
    """
    Get digital twin instance for a city.
    
    Args:
        city_key: City identifier
        
    Returns:
        DigitalTwin instance
        
    Raises:
        HTTPException if twin not initialized
    """
    if city_key not in digital_twins:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin for '{city_key}' not initialized. "
                   f"Call /api/twin/{city_key}/initialize first."
        )
    return digital_twins[city_key]


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/cities")
async def list_cities():
    """
    List all available cities with their digital twin status.
    """
    cities = []
    
    for city_key, config in CITIES_CONFIG.items():
        status = "not_initialized"
        metadata = {}
        
        if city_key in digital_twins:
            twin = digital_twins[city_key]
            metadata = twin.get_metadata()
            status = metadata.get("status", "unknown")
        
        cities.append({
            "key": city_key,
            "name": config["name"],
            "display_name": config["display_name"],
            "center": config["center"],
            "zoom_level": config["zoom_level"],
            "status": status,
            "metadata": metadata,
        })
    
    return {
        "cities": cities,
        "total_cities": len(cities),
    }


@router.get("/{city_key}/initialize")
async def initialize_twin(
    city_key: str,
    force_rebuild: bool = Query(False, description="Force rebuild even if cache exists")
):
    """
    Initialize or rebuild digital twin for a city.
    
    This will:
    1. Download road network (if needed)
    2. Map accidents to segments
    3. Calculate risk scores
    4. Generate heatmaps
    
    First time: 5-10 minutes
    From cache: <10 seconds
    """
    if city_key not in CITIES_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city_key}' not found. "
                   f"Available: {list(CITIES_CONFIG.keys())}"
        )
    
    try:
        logger.info(f"Initializing digital twin for {city_key}...")
        
        # Create twin instance
        # Try to get predictor from global scope (if loaded)
        predictor = None
        try:
            from main import predictor_instance
            predictor = predictor_instance
            logger.info("Using ML predictor for risk calculation")
        except ImportError:
            logger.warning(
                "ML predictor not available. "
                "Risk calculation will use historical data only."
            )
        
        twin = DigitalTwin(city_key, predictor=predictor)
        
        # Build or load
        metadata = twin.initialize_twin(force_rebuild=force_rebuild)
        
        # Store in global registry
        digital_twins[city_key] = twin
        
        logger.info(f"Digital twin ready for {city_key}")
        
        return {
            "status": "success",
            "city": city_key,
            "metadata": metadata,
            "message": "Digital twin initialized successfully",
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize twin for {city_key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Twin initialization failed: {str(e)}"
        )


@router.get("/{city_key}/metadata")
async def get_twin_metadata(city_key: str):
    """
    Get metadata for a digital twin.
    """
    twin = get_twin(city_key)
    
    return {
        "city": city_key,
        "metadata": twin.get_metadata(),
        "stats": twin.get_stats(),
    }


@router.get("/{city_key}/heatmap")
async def get_heatmap(
    city_key: str,
    type: str = Query("segments", description="Heatmap type: 'grid' or 'segments'"),
    risk_threshold: float = Query(0, description="Minimum risk threshold (0-100)")
):
    """
    Get heatmap data for visualization.
    """
    twin = get_twin(city_key)

    if type not in ["grid", "segments"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid type. Use 'grid' or 'segments'"
        )

    try:
        heatmap_data = twin.get_heatmap_api_response(
            heatmap_type=type,
            risk_threshold=risk_threshold
        )

        # ── NORMALIZE for frontend compatibility ──
        if isinstance(heatmap_data, dict) and "data" in heatmap_data:
            normalized = []
            for seg in heatmap_data["data"]:
                if not isinstance(seg, dict):
                    normalized.append(seg)
                    continue
                # Fix 1: ensure risk_score field exists
                if "risk_score" not in seg or seg.get("risk_score", 0) == 0:
                    seg["risk_score"] = seg.get("composite_risk", seg.get("risk_score", 0))
                # Fix 2: ensure centroid array exists for circle markers
                if "centroid" not in seg or seg["centroid"] is None:
                    lat = seg.get("centroid_lat")
                    lon = seg.get("centroid_lon")
                    if lat is not None and lon is not None:
                        seg["centroid"] = [lat, lon]
                normalized.append(seg)
            heatmap_data["data"] = normalized

        return heatmap_data

    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate heatmap: {str(e)}"
        )


@router.get("/{city_key}/segments/top-dangerous")
async def get_top_dangerous_segments(
    city_key: str,
    limit: int = Query(10, ge=1, le=100, description="Number of segments to return"),
    min_risk: float = Query(60, ge=0, le=100, description="Minimum risk score")
):
    """
    Get most dangerous road segments.
    
    Returns segments sorted by risk score (highest first).
    """
    twin = get_twin(city_key)
    
    try:
        segments = twin.get_top_dangerous_segments(n=limit, min_risk=min_risk)
        
        return {
            "city": city_key,
            "segments": segments,
            "count": len(segments),
            "min_risk": min_risk,
        }
        
    except Exception as e:
        logger.error(f"Top dangerous query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/{city_key}/segment/{segment_id}")
async def get_segment_details(city_key: str, segment_id: str):
    """
    Get detailed information for a specific road segment.
    
    Includes:
    - Road name, type, length
    - Risk score and category
    - Accident history
    - Weather/time distributions
    """
    twin = get_twin(city_key)
    
    try:
        segment_info = twin.get_segment_info(segment_id)
        
        if segment_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Segment '{segment_id}' not found"
            )
        
        return {
            "city": city_key,
            "segment": segment_info,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segment query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/{city_key}/segment/{segment_id}/simulate")
async def simulate_segment_scenario(
    city_key: str,
    segment_id: str,
    scenario_type: str = Query(..., description="Scenario type: weather, time, traffic, intervention"),
    weather: Optional[str] = Query(None, description="Weather condition"),
    time_period: Optional[str] = Query(None, description="Time period"),
    traffic_level: Optional[str] = Query(None, description="Traffic level"),
    intervention_id: Optional[str] = Query(None, description="Intervention ID"),
):
    """
    Simulate a scenario on a road segment.
    
    Scenario types:
    - weather: Simulate different weather (requires weather param)
    - time: Simulate different time of day (requires time_period param)
    - traffic: Simulate different traffic level (requires traffic_level param)
    - intervention: Simulate safety intervention (requires intervention_id param)
    """
    twin = get_twin(city_key)
    
    if not twin.scenario_simulator:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator not available (ML predictor required)"
        )
    
    try:
        kwargs = {}
        if weather:
            kwargs["weather"] = weather
        if time_period:
            kwargs["time_period"] = time_period
        if traffic_level:
            kwargs["traffic_level"] = traffic_level
        if intervention_id:
            kwargs["intervention_id"] = intervention_id
        
        result = twin.simulate_scenario(
            segment_id,
            scenario_type,
            **kwargs
        )
        
        return {
            "city": city_key,
            "segment_id": segment_id,
            "scenario_type": scenario_type,
            "result": result,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


@router.post("/{city_key}/refresh")
async def refresh_twin(city_key: str):
    """
    Rebuild digital twin with latest data.
    
    This will re-download road network, re-map accidents,
    and recalculate all risks.
    
    Warning: Takes 5-10 minutes!
    """
    if city_key not in CITIES_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city_key}' not found"
        )
    
    try:
        logger.info(f"Refreshing digital twin for {city_key}...")
        
        # Get or create twin
        if city_key in digital_twins:
            twin = digital_twins[city_key]
        else:
            predictor = None
            try:
                from main import predictor_instance
                predictor = predictor_instance
            except ImportError:
                pass
            
            twin = DigitalTwin(city_key, predictor=predictor)
            digital_twins[city_key] = twin
        
        # Rebuild
        metadata = twin.refresh_twin()
        
        return {
            "status": "success",
            "city": city_key,
            "metadata": metadata,
            "message": "Digital twin refreshed successfully",
        }
        
    except Exception as e:
        logger.error(f"Refresh failed for {city_key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Refresh failed: {str(e)}"
        )


@router.get("/{city_key}/stats")
async def get_twin_stats(city_key: str):
    """
    Get comprehensive statistics for a digital twin.
    """
    twin = get_twin(city_key)
    
    try:
        stats = twin.get_stats()
        
        return {
            "city": city_key,
            "stats": stats,
            "generated_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
