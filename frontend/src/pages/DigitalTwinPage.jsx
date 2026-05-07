import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, Popup, useMap } from 'react-leaflet';
import { motion } from 'framer-motion';
import {
  FiMap, FiRefreshCw, FiAlertTriangle,
  FiActivity, FiInfo, FiDatabase, FiLayers
} from 'react-icons/fi';
import toast from 'react-hot-toast';

import { api } from '../api/apiClient';
import { CITY_CENTERS } from '../utils/constants';
import LoadingSpinner from '../components/common/LoadingSpinner';
import KPICard from '../components/common/KPICard';
import CitySelector from '../components/digital_twin/CitySelector';
import StatisticsPanel from '../components/digital_twin/StatisticsPanel';
import TopDangerousList from '../components/digital_twin/TopDangerousList';
import SegmentDetailPanel from '../components/digital_twin/SegmentDetailPanel';
import RiskLegend from '../components/digital_twin/RiskLegend';

// ─── Performance cap ───────────────────────────────────────────────────────────
const MAX_MAP_SEGMENTS = 800;

// ─── Vibrant Color scheme per user requirements ────────────────────────────────
// 0-10%   = Green   #22C55E  "Zero Accidents"
// 10-40%  = Blue    #3B82F6  "Low Risk"
// 40-60%  = Yellow  #EAB308  "Moderate Risk"
// 60-80%  = Orange  #F97316  "High Risk"
// 80-95%+ = Red     #EF4444  "Very High Risk"

const RISK_LEVELS = [
  { min: 0,  max: 10,  color: '#22C55E', label: 'Zero Accidents', shortLabel: 'Zero',   emoji: '🟢' },
  { min: 10, max: 40,  color: '#3B82F6', label: 'Low Risk',       shortLabel: 'Low',    emoji: '🔵' },
  { min: 40, max: 60,  color: '#EAB308', label: 'Moderate Risk',   shortLabel: 'Mod',    emoji: '🟡' },
  { min: 60, max: 80,  color: '#F97316', label: 'High Risk',       shortLabel: 'High',   emoji: '🟠' },
  { min: 80, max: 100, color: '#EF4444', label: 'Very High Risk',  shortLabel: 'V.High', emoji: '🔴' },
];

const getSegmentColor = (seg) => {
  const score = seg.risk_score ?? 0;
  if (score >= 80) return '#EF4444';   // Red - Very High Risk (80-95%+)
  if (score >= 60) return '#F97316';   // Orange - High Risk (60-80%)
  if (score >= 40) return '#EAB308';   // Yellow - Moderate Risk (40-60%)
  if (score >= 10) return '#3B82F6';   // Blue - Low Risk (10-40%)
  return '#22C55E';                     // Green - Zero Accidents (0-10%)
};

const getSegmentWeight = (seg) => {
  if (seg.weight) return Math.min(seg.weight, 6);
  const score = seg.risk_score ?? 0;
  if (score >= 80) return 6;
  if (score >= 60) return 5;
  if (score >= 40) return 4;
  if (score >= 10) return 3;
  return 2;
};


const getGlowWeight = (seg) => {
  const base = getSegmentWeight(seg);
  return base + 8;
};

const getGlowOpacity = (seg) => {
  const score = seg.risk_score ?? 0;
  if (score >= 80) return 0.85;
  if (score >= 60) return 0.75;
  if (score >= 40) return 0.65;
  if (score >= 10) return 0.55;
  return 0.40;
};

const getRiskCategory = (score) => {
  if (score >= 80) return 'Very High Risk';
  if (score >= 60) return 'High Risk';
  if (score >= 40) return 'Moderate Risk';
  if (score >= 10) return 'Low Risk';
  return 'Zero Accidents';
};

// ─── Compute centroid from polyline coordinates ────────────────────────────────
const computeCentroid = (coords) => {
  if (!coords || !Array.isArray(coords) || coords.length === 0) return null;
  let latSum = 0, lngSum = 0, count = 0;
  for (const pt of coords) {
    if (Array.isArray(pt) && pt.length >= 2 && typeof pt[0] === 'number' && typeof pt[1] === 'number') {
      latSum += pt[0];
      lngSum += pt[1];
      count++;
    }
  }
  if (count === 0) return null;
  return [latSum / count, lngSum / count];
};

// ─── Compute circle marker radius based on accident count ──────────────────────
const getCircleRadius = (seg) => {
  const accidents = seg.total_accidents ?? seg.accident_count ?? 0;
  if (accidents === 0) return 4;
  // Scale: min 5, max 18 for visual clarity
  const radius = Math.max(5, Math.min(18, 5 + Math.log2(accidents + 1) * 2.5));
  return radius;
};

// ─── Risk distribution computation ─────────────────────────────────────────────
const computeRiskDistribution = (segments) => {
  const dist = {
    zero:  { count: 0, color: '#22C55E', label: 'Zero Accidents', range: '0-10%' },
    low:   { count: 0, color: '#3B82F6', label: 'Low Risk',       range: '10-40%' },
    mod:   { count: 0, color: '#EAB308', label: 'Moderate Risk',   range: '40-60%' },
    high:  { count: 0, color: '#F97316', label: 'High Risk',       range: '60-80%' },
    vhigh: { count: 0, color: '#EF4444', label: 'Very High Risk',  range: '80-95%+' },
  };

  for (const seg of segments) {
    const score = seg.risk_score ?? 0;
    if (score >= 80) dist.vhigh.count++;
    else if (score >= 60) dist.high.count++;
    else if (score >= 40) dist.mod.count++;
    else if (score >= 10) dist.low.count++;
    else dist.zero.count++;
  }

  return dist;
};

// ─── Map recenter ──────────────────────────────────────────────────────────────
const MapRecenter = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    if (center && map) {
      try { map.setView([center.lat, center.lng], zoom || 12); }
      catch (e) { /* ignore */ }
    }
  }, [center, zoom, map]);
  return null;
};

// ─── Map Legend Overlay (positioned on the map) ────────────────────────────────
const MapLegendOverlay = () => (
  <div
    style={{
      position: 'absolute',
      bottom: 20,
      left: 20,
      zIndex: 1000,
      background: 'rgba(15, 23, 42, 0.92)',
      backdropFilter: 'blur(8px)',
      border: '1px solid rgba(148, 163, 184, 0.2)',
      borderRadius: 12,
      padding: '12px 16px',
      pointerEvents: 'auto',
    }}
  >
    <p style={{ fontSize: 11, fontWeight: 600, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em', margin: '0 0 8px 0' }}>
      Accident Risk Level
    </p>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      {RISK_LEVELS.map((level) => (
        <div key={level.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 16,
              height: 8,
              borderRadius: 3,
              backgroundColor: level.color,
              flexShrink: 0,
              boxShadow: `0 0 6px ${level.color}60`,
            }}
          />
          <span style={{ fontSize: 11, color: '#E2E8F0', whiteSpace: 'nowrap' }}>
            {level.label}
          </span>
          <span style={{ fontSize: 10, color: '#64748B', marginLeft: 'auto', fontFamily: 'monospace' }}>
            {level.min === 0 ? '0' : level.min}-{level.max}%
          </span>
        </div>
      ))}
    </div>
    <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid rgba(148, 163, 184, 0.15)' }}>
      <p style={{ fontSize: 9, color: '#4ADE80', margin: 0 }}>
        ✓ Based on REAL Delhi Police Data
      </p>
      <p style={{ fontSize: 9, color: '#64748B', margin: '2px 0 0 0' }}>
        Circle size = accident count
      </p>
    </div>
  </div>
);

// ─── Main Page ────────────────────────────────────────────────────────────────
const DigitalTwinPage = () => {
  const [cities,           setCities]           = useState([]);
  const [selectedCity,     setSelectedCity]     = useState(null);
  const [initializingCity, setInitializingCity] = useState(null);
  const [segments,         setSegments]         = useState([]);
  const [topDangerous,     setTopDangerous]     = useState([]);
  const [stats,            setStats]            = useState(null);
  const [metadata,         setMetadata]         = useState(null);
  const [selectedSegment,  setSelectedSegment]  = useState(null);
  const [detailSegment,    setDetailSegment]    = useState(null);
  const [loading,          setLoading]          = useState(false);
  const [mapLoading,       setMapLoading]       = useState(false);
  const [riskFilter,       setRiskFilter]       = useState('all');
  const [sidebarTab,       setSidebarTab]       = useState('stats');
  const [mapKey,           setMapKey]           = useState('map-initial');
  const [mapReady,         setMapReady]         = useState(false);

  // ── load cities ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const fetchCities = async () => {
      try {
        const res = await api.twinCities();
        const cityList = res.data?.cities || [];
        setCities(cityList);
        const ready = cityList.find((c) => c.status === 'ready');
        if (ready) {
          setSelectedCity(ready.key);
          loadCityData(ready.key);
        }
      } catch (err) {
        toast.error('Failed to load cities');
        console.error(err);
      }
    };
    fetchCities();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── load city data ───────────────────────────────────────────────────────────
  const loadCityData = useCallback(async (cityKey) => {
    if (!cityKey) return;
    setMapLoading(true);
    setMapReady(false);
    setSegments([]);
    setTopDangerous([]);
    setStats(null);
    setMetadata(null);
    setDetailSegment(null);
    setSelectedSegment(null);

    try {
      const [heatmapRes, statsRes, topRes] = await Promise.allSettled([
        api.twinHeatmap(cityKey, 'segments', 0),
        api.twinStats(cityKey),
        api.twinTopDangerous(cityKey, 15, 0),
      ]);

      if (heatmapRes.status === 'fulfilled') {
        const allSegs = heatmapRes.value.data?.data || [];
        setSegments(allSegs);
        console.log(`Loaded ${allSegs.length} segments for ${cityKey}`);
      }
      if (statsRes.status === 'fulfilled') {
        const d = statsRes.value.data;
        setStats(d.stats);
        setMetadata(d.stats?.metadata);
      }
      if (topRes.status === 'fulfilled') {
        setTopDangerous(topRes.value.data?.segments || []);
      }
    } catch (err) {
      toast.error('Failed to load city data');
      console.error(err);
    } finally {
      setMapLoading(false);
      setTimeout(() => setMapReady(true), 300);
    }
  }, []);

  // ── initialize twin ──────────────────────────────────────────────────────────
  const handleInitialize = useCallback(async (cityKey) => {
    setInitializingCity(cityKey);
    toast('Building digital twin with REAL Delhi data... this may take 2-5 minutes.', { icon: '⏳', duration: 8000 });
    try {
      const res = await api.twinInitialize(cityKey, true);
      if (res.data?.status === 'success') {
        toast.success(`Digital twin for ${cityKey} is ready with real Delhi data!`);
        const citiesRes = await api.twinCities();
        setCities(citiesRes.data?.cities || []);
        setSelectedCity(cityKey);
        setMapKey(`map-${cityKey}-${Date.now()}`);
        await loadCityData(cityKey);
      }
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Initialization failed');
    } finally {
      setInitializingCity(null);
    }
  }, [loadCityData]);

  // ── select city ──────────────────────────────────────────────────────────────
  const handleCitySelect = useCallback((cityKey) => {
    if (cityKey === selectedCity) return;
    setSelectedCity(cityKey);
    setSelectedSegment(null);
    setDetailSegment(null);
    setSidebarTab('stats');
    setMapReady(false);
    setMapKey(`map-${cityKey}-${Date.now()}`);
    loadCityData(cityKey);
  }, [loadCityData, selectedCity]);

  // ── segment detail ───────────────────────────────────────────────────────────
  const fetchSegmentDetail = useCallback(async (seg) => {
    setSelectedSegment(seg.segment_id);
    setSidebarTab('detail');
    if (selectedCity) {
      try {
        const res = await api.twinSegmentDetails(selectedCity, seg.segment_id);
        setDetailSegment(res.data?.segment || seg);
      } catch {
        setDetailSegment(seg);
      }
    } else {
      setDetailSegment(seg);
    }
  }, [selectedCity]);

  // ── refresh ──────────────────────────────────────────────────────────────────
  const handleRefresh = useCallback(async () => {
    if (!selectedCity) return;
    setLoading(true);
    try {
      await api.twinRefresh(selectedCity);
      toast.success('Twin refreshed with real data!');
      setMapKey(`map-${selectedCity}-${Date.now()}`);
      await loadCityData(selectedCity);
    } catch {
      toast.error('Refresh failed');
    } finally {
      setLoading(false);
    }
  }, [selectedCity, loadCityData]);

  // ── filtered + capped segments for map ───────────────────────────────────────
  const filteredSegments = useMemo(() => {
    const filtered = segments.filter((s) => {
      const score = s.risk_score ?? 0;

      switch (riskFilter) {
        case 'zero':
          return score >= 0 && score < 10;
        case 'low':
          return score >= 10 && score < 40;
        case 'moderate':
          return score >= 40 && score < 60;
        case 'high':
          return score >= 60 && score < 80;
        case 'vhigh':
          return score >= 80;
        default:
          return true; // all
      }
    });

    if (selectedSegment) {
      const sel = segments.find((s) => s.segment_id === selectedSegment);
      if (sel && !filtered.find((s) => s.segment_id === sel.segment_id)) {
        return [sel, ...filtered].slice(0, MAX_MAP_SEGMENTS);
      }
    }
    return filtered.slice(0, MAX_MAP_SEGMENTS);
  }, [segments, riskFilter, selectedSegment]);

  const totalFiltered = useMemo(
    () => {
      return segments.filter((s) => {
        const score = s.risk_score ?? 0;

        switch (riskFilter) {
          case 'zero':
            return score >= 0 && score < 10;
          case 'low':
            return score >= 10 && score < 40;
          case 'moderate':
            return score >= 40 && score < 60;
          case 'high':
            return score >= 60 && score < 80;
          case 'vhigh':
            return score >= 80;
          default:
            return true;
        }
      }).length;
    },
    [segments, riskFilter]
  );

  // ── Risk distribution for stats ──────────────────────────────────────────────
  const riskDistribution = useMemo(() => computeRiskDistribution(segments), [segments]);

  // city center from backend data
  const cityConfig   = cities.find((c) => c.key === selectedCity);
  const cityCenter   = cityConfig?.center
    ? { lat: cityConfig.center[0], lng: cityConfig.center[1] }
    : CITY_CENTERS.delhi;
  const cityZoom     = cityConfig?.zoom_level || 12;

  // KPIs
  const riskStats    = stats?.risk_statistics    || {};
  const networkStats = stats?.network_statistics  || {};
  const accStats     = stats?.accident_statistics || {};

  // Computed stats for real data display
  const totalAccidentsFromSegs = useMemo(
    () => segments.reduce((sum, s) => sum + (s.total_accidents ?? s.accident_count ?? 0), 0),
    [segments]
  );

  return (
    <div className="space-y-5">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between flex-wrap gap-3"
      >
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FiMap className="text-pbi-blue" />
            Digital Twin — Delhi Road Network
          </h1>
          <p className="text-sm text-pbi-muted mt-0.5">
            Real Delhi Police accident data mapped to road segments
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selectedCity && (
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                         bg-pbi-bg2 border border-pbi-border text-pbi-text2 hover:text-white
                         transition-all duration-200 disabled:opacity-50"
            >
              <FiRefreshCw className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          )}
        </div>
      </motion.div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard icon={<FiMap />}          value={networkStats.total_edges?.toLocaleString() ?? segments.length.toLocaleString() ?? '—'} label="Road Segments"    color="#3B82F6" delay={0}    />
        <KPICard icon={<FiAlertTriangle />} value={accStats.total_accidents?.toLocaleString() ?? totalAccidentsFromSegs.toLocaleString()} label="Total Accidents"  color="#EF4444" delay={0.05} />
        <KPICard icon={<FiActivity />}      value={riskStats.mean_risk != null ? `${riskStats.mean_risk.toFixed(1)}%` : '—'}             label="Avg Risk Score"   color="#EAB308" delay={0.1}  />
        <KPICard icon={<FiAlertTriangle />} value={riskStats.high_risk_count?.toLocaleString() ?? '—'}                                   label="High Risk Segs"   color="#F97316" delay={0.15} subtext="Risk >= 60%" />
      </div>

      {/* Filter Controls */}
      <div className="glass-card-static p-4 rounded-xl">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <label className="text-xs text-pbi-muted whitespace-nowrap">
              Filter: <span className="text-white font-bold">
                {riskFilter === 'all' ? 'All' : riskFilter.toUpperCase()}
              </span>
            </label>
          </div>

          {/* Exact category filter buttons */}
          <div className="flex gap-1.5 flex-wrap">
            {[
              { label: 'All',       value: 'all',      col: 'text-pbi-muted'   },
              { label: 'Zero',      value: 'zero',     col: 'text-green-400'   },
              { label: 'Low',       value: 'low',      col: 'text-blue-400'    },
              { label: 'Moderate',  value: 'moderate', col: 'text-yellow-400'  },
              { label: 'High',      value: 'high',     col: 'text-orange-400'  },
              { label: 'V.High',    value: 'vhigh',    col: 'text-red-400'     },
            ].map((f) => (
              <button
                key={f.value}
                onClick={() => setRiskFilter(f.value)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-all
                            ${riskFilter === f.value
                              ? `${f.col} border-current bg-white/5`
                              : 'border-pbi-border text-pbi-muted hover:text-white bg-pbi-bg2'}`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Segment count */}
          <div className="ml-auto text-xs text-pbi-muted">
            {segments.length > 0 ? (
              <>
                Rendering{' '}
                <span className="text-white font-bold">{filteredSegments.length}</span>
                {totalFiltered > MAX_MAP_SEGMENTS && (
                  <span className="text-pbi-yellow"> / {MAX_MAP_SEGMENTS} cap</span>
                )}
                {' '}of{' '}
                <span className="text-white font-bold">{totalFiltered}</span> matching
                {' '}({segments.length.toLocaleString()} total)
              </>
            ) : (
              <span>No segments loaded</span>
            )}
          </div>
        </div>

        {/* Color scale - vibrant colors per spec */}
        <div className="flex items-center gap-3 mt-3 flex-wrap">
          {RISK_LEVELS.map((l) => (
            <div key={l.label} className="flex items-center gap-1.5">
              <div
                className="w-6 h-2 rounded-sm"
                style={{ backgroundColor: l.color, boxShadow: `0 0 4px ${l.color}50` }}
              />
              <span className="text-[10px] text-pbi-text2">{l.emoji} {l.label} ({l.min}-{l.max}%)</span>
            </div>
          ))}
        </div>

        {/* Data source badge */}
        <div className="mt-2 flex items-center gap-2">
          <span className="text-[10px] text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full">
            ✓ REAL Delhi Police Data
          </span>
          <span className="text-[10px] text-pbi-muted">
            Source: delhiDatasets (2016-2024)
          </span>
        </div>

        {/* Performance warning */}
        {riskFilter === 'all' && segments.length > 1000 && (
          <p className="mt-2 text-xs text-pbi-yellow flex items-center gap-1">
            ⚡ Showing top {MAX_MAP_SEGMENTS} of {segments.length.toLocaleString()} segments.
            Use filters above to see specific risk levels.
          </p>
        )}
      </div>

      {/* Map + Sidebar */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-5">

        {/* MAP */}
        <div className="glass-card-static rounded-xl overflow-hidden" style={{ height: 560 }}>
          {mapLoading ? (
            <div className="flex flex-col items-center justify-center h-full gap-3">
              <LoadingSpinner text="Loading road segments..." />
              <p className="text-xs text-pbi-muted">Processing {selectedCity} real accident data</p>
            </div>
          ) : !selectedCity ? (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <FiMap className="text-5xl text-pbi-muted" />
              <p className="text-white font-semibold">Select a city to view road network</p>
            </div>
          ) : !mapReady ? (
            <div className="flex items-center justify-center h-full">
              <LoadingSpinner text="Preparing map..." />
            </div>
          ) : (
            <div style={{ position: 'relative', height: '100%', width: '100%' }}>
              <MapContainer
                key={mapKey}
                center={[cityCenter.lat, cityCenter.lng]}
                zoom={cityZoom}
                style={{ height: '100%', width: '100%' }}
                preferCanvas={false}
                zoomAnimation={false}
                markerZoomAnimation={false}
                updateWhenZooming={false}
                updateWhenIdle={true}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  maxZoom={19}
                />
                <MapRecenter center={cityCenter} zoom={cityZoom} />

                
{filteredSegments.map((seg, idx) => {
  const coords = Array.isArray(seg.coordinates)
    ? seg.coordinates.filter((pt) =>
        Array.isArray(pt) &&
        pt.length >= 2 &&
        typeof pt[0] === 'number' &&
        typeof pt[1] === 'number'
      )
    : [];

  const color = getSegmentColor(seg);
  const weight = getSegmentWeight(seg);
  const isSelected = selectedSegment === seg.segment_id;
  const riskScore = seg.risk_score ?? 0;
  const category = seg.risk_category || getRiskCategory(riskScore);
  const accidents = seg.total_accidents ?? seg.accident_count ?? 0;

  // Fallback line if coordinates are somehow missing
  const centroid = seg.centroid
    ? (Array.isArray(seg.centroid)
        ? seg.centroid
        : [seg.centroid.lat ?? seg.centroid[0], seg.centroid.lng ?? seg.centroid[1]])
    : computeCentroid(coords);

  const displayCoords = coords.length >= 2
    ? coords
    : (centroid
        ? [
            [centroid[0] - 0.0002, centroid[1] - 0.0002],
            [centroid[0] + 0.0002, centroid[1] + 0.0002],
          ]
        : []);

  const popupContent = (
    <div style={{ minWidth: 200, fontFamily: 'sans-serif' }}>
      <p style={{ fontWeight: 700, fontSize: 13, margin: '0 0 3px', color: '#111' }}>
        {seg.road_name || seg.name || 'Unnamed Road'}
      </p>
      <p style={{ fontSize: 11, color: '#666', margin: '0 0 6px' }}>
        {seg.road_type || seg.road_category || 'Unknown type'}
      </p>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: '#666' }}>Risk Score</span>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>
          {riskScore.toFixed(1)}%
        </span>
      </div>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: '#666' }}>Category</span>
        <span style={{ fontSize: 11, fontWeight: 600, color }}>
          {category}
        </span>
      </div>
      {accidents > 0 && (
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 3 }}>
          <span style={{ fontSize: 11, color: '#666' }}>Accidents</span>
          <span style={{ fontSize: 11, fontWeight: 600, color: '#333' }}>
            {accidents}
          </span>
        </div>
      )}
      {seg.length_m != null && (
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 3 }}>
          <span style={{ fontSize: 11, color: '#666' }}>Length</span>
          <span style={{ fontSize: 11, color: '#333' }}>
            {(seg.length_m / 1000).toFixed(2)} km
          </span>
        </div>
      )}
      {seg.segment_id && (
        <div style={{ display:'flex', justifyContent:'space-between' }}>
          <span style={{ fontSize: 11, color: '#666' }}>Segment ID</span>
          <span style={{ fontSize: 10, color: '#999', fontFamily: 'monospace' }}>
            {seg.segment_id}
          </span>
        </div>
      )}
    </div>
  );

  return (
    <React.Fragment key={seg.segment_id || `seg-${idx}`}>
      {displayCoords.length >= 2 && (
        <>
          {/* soft glow */}
          <Polyline
            positions={displayCoords}
            pathOptions={{
              color,
              weight: Math.max(6, weight + 6),
              opacity: isSelected ? 0.35 : 0.18,
              lineCap: 'round',
              lineJoin: 'round',
            }}
          />
          {/* main road line */}
          <Polyline
            positions={displayCoords}
            pathOptions={{
              color: isSelected ? '#FFFFFF' : color,
              weight: isSelected ? Math.max(5, weight + 2) : Math.max(3, weight),
              opacity: 0.98,
              lineCap: 'round',
              lineJoin: 'round',
            }}
            eventHandlers={{
              click: () => fetchSegmentDetail(seg),
            }}
          >
            <Popup>{popupContent}</Popup>
          </Polyline>
        </>
      )}
    </React.Fragment>
  );
})}

              </MapContainer>

              {/* Map Legend Overlay */}
              <MapLegendOverlay />
            </div>
          )}
        </div>

        {/* SIDEBAR */}
        <div className="flex flex-col gap-4">
          <CitySelector
            cities={cities}
            selectedCity={selectedCity}
            onSelect={handleCitySelect}
            onInitialize={handleInitialize}
            initializingCity={initializingCity}
          />

          {selectedCity && (
            <div className="flex gap-1 bg-pbi-bg2 rounded-lg p-1">
              {[
                { id: 'stats',  label: 'Stats',   icon: <FiActivity /> },
                { id: 'top',    label: 'Top Risk', icon: <FiAlertTriangle /> },
                { id: 'detail', label: 'Segment',  icon: <FiInfo /> },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSidebarTab(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2
                              rounded-md text-xs font-medium transition-all duration-200
                              ${sidebarTab === tab.id
                                ? 'bg-pbi-blue text-white shadow'
                                : 'text-pbi-muted hover:text-white'}`}
                >
                  {tab.icon}
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>
          )}

          {selectedCity && (
            <>
              {sidebarTab === 'stats' && (
                <>
                  <StatisticsPanel stats={stats} metadata={metadata} />

                  {/* Risk Distribution Panel — 5 categories */}
                  <div className="glass-card-static p-4 rounded-xl">
                    <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                      <FiLayers className="text-pbi-blue" />
                      Risk Distribution
                    </h4>
                    {segments.length > 0 ? (
                      <>
                        {/* Distribution bar */}
                        <div className="flex h-3 rounded-full overflow-hidden gap-0.5 mb-2">
                          {Object.values(riskDistribution).map((cat) => {
                            const pct = segments.length > 0 ? (cat.count / segments.length) * 100 : 0;
                            return pct > 0 ? (
                              <div
                                key={cat.label}
                                className="h-full transition-all"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor: cat.color,
                                  minWidth: pct > 0 ? 2 : 0,
                                }}
                                title={`${cat.label}: ${cat.count} (${pct.toFixed(1)}%)`}
                              />
                            ) : null;
                          })}
                        </div>

                        {/* Category breakdown */}
                        <div className="space-y-2 mt-3">
                          {Object.values(riskDistribution).map((cat) => {
                            const pct = segments.length > 0 ? (cat.count / segments.length) * 100 : 0;
                            return (
                              <div key={cat.label} className="flex items-center gap-2">
                                <div
                                  className="w-3 h-3 rounded-sm flex-shrink-0"
                                  style={{ backgroundColor: cat.color }}
                                />
                                <span className="text-xs text-white flex-1">{cat.label}</span>
                                <span className="text-xs text-pbi-text2 font-mono">
                                  {cat.count.toLocaleString()}
                                </span>
                                <span className="text-[10px] text-pbi-muted font-mono w-12 text-right">
                                  {pct.toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-pbi-muted text-center py-3">
                        Load segments to see distribution
                      </p>
                    )}
                  </div>

                  {/* Real Data Stats Panel */}
                  <div className="glass-card-static p-4 rounded-xl">
                    <h4 className="text-xs font-semibold text-pbi-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                      <FiDatabase className="text-emerald-400" />
                      Data Overview
                    </h4>
                    <div className="space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-pbi-muted">Total Segments Mapped</span>
                        <span className="text-xs text-white font-bold">
                          {segments.length.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-pbi-muted">Total Accidents</span>
                        <span className="text-xs text-white font-bold">
                          {(accStats.total_accidents ?? totalAccidentsFromSegs)?.toLocaleString?.() ?? '—'}
                        </span>
                      </div>
                      {accStats.mapped_accidents != null && (
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-pbi-muted">Mapped Accidents</span>
                          <span className="text-xs text-white font-bold">
                            {accStats.mapped_accidents.toLocaleString()}
                          </span>
                        </div>
                      )}
                      {accStats.match_rate != null && (
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-pbi-muted">Match Rate</span>
                          <span className="text-xs text-emerald-400 font-bold">
                            {(accStats.match_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {/* Dataset breakdown from metadata */}
                      {metadata?.dataset_breakdown && (
                        <div className="pt-2 mt-1 border-t border-pbi-border">
                          <p className="text-[10px] text-pbi-muted uppercase tracking-wider mb-1.5">Dataset Breakdown</p>
                          {Object.entries(metadata.dataset_breakdown).map(([dsKey, dsVal]) => (
                            <div key={dsKey} className="flex items-center justify-between py-0.5">
                              <span className="text-xs text-pbi-text2">{dsKey}</span>
                              <span className="text-xs text-white font-mono">
                                {typeof dsVal === 'number' ? dsVal.toLocaleString() : String(dsVal)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                      {/* Virtual vs Real segment counts */}
                      {metadata && (metadata.virtual_segments != null || metadata.real_segments != null) && (
                        <div className="pt-2 mt-1 border-t border-pbi-border">
                          <p className="text-[10px] text-pbi-muted uppercase tracking-wider mb-1.5">Segment Origin</p>
                          {metadata.real_segments != null && (
                            <div className="flex items-center justify-between py-0.5">
                              <span className="text-xs text-blue-400">Real Segments</span>
                              <span className="text-xs text-white font-mono">
                                {metadata.real_segments.toLocaleString()}
                              </span>
                            </div>
                          )}
                          {metadata.virtual_segments != null && (
                            <div className="flex items-center justify-between py-0.5">
                              <span className="text-xs text-purple-400">Virtual Segments</span>
                              <span className="text-xs text-white font-mono">
                                {metadata.virtual_segments.toLocaleString()}
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  <RiskLegend />
                </>
              )}
              {sidebarTab === 'top' && (
                <TopDangerousList
                  segments={topDangerous}
                  onSelectSegment={fetchSegmentDetail}
                  selectedSegmentId={selectedSegment}
                  loading={mapLoading}
                />
              )}
              {sidebarTab === 'detail' && (
                <SegmentDetailPanel
                  segment={detailSegment}
                  onClose={() => {
                    setDetailSegment(null);
                    setSelectedSegment(null);
                    setSidebarTab('stats');
                  }}
                  onSimulate={() => {
                    toast('Go to Simulation page to run scenarios', { icon: '🔬' });
                  }}
                />
              )}
            </>
          )}

          {!selectedCity && (
            <div className="glass-card-static rounded-xl p-6 text-center">
              <FiMap className="text-3xl text-pbi-muted mx-auto mb-3" />
              <p className="text-sm text-white font-medium">Select a city to begin</p>
              <p className="text-xs text-pbi-muted mt-1">
                Initialize a digital twin to view the risk heatmap
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DigitalTwinPage;
